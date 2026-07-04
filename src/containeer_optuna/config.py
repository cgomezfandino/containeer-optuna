"""Configuration system for containeer-optuna.

Defines pydantic models for experiments, datasets, models, optimization, and
cross-validation, plus YAML loaders for the registry files under ``config/``.

All path resolution is anchored to the repository root (the parent of the
``src`` directory) so loaders work regardless of the current working directory.
The base directory can also be overridden through :class:`Settings` (``CO_DIR``
env var) or by passing ``base_dir`` explicitly to the loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: this file lives at <repo>/src/containeer_optuna/config.py,
# so three `.parent` calls bring us back to the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Allowed kinds of model entries in ``models.yaml``. ``classification`` is
# included forward-compatibly for the M4 milestone; it is not yet used.
ModelType = Literal["regression", "clustering", "reducer", "scaler", "classification"]

# Allowed experiment tasks. M0 supports regression and clustering; M4 will add
# classification, hence it is accepted at the config layer already.
TaskType = Literal["regression", "clustering", "classification"]

# Pluggable regression optimization metrics (M1).
# - r2: coefficient of determination (higher is better, maximize).
# - mse / rmse / mae: error metrics (lower is better, minimize).
RegressionMetric = Literal["r2", "mse", "rmse", "mae"]

# Metric → optimization direction map (M1).
METRIC_DIRECTION: dict[str, str] = {
    "r2": "maximize",
    "mse": "minimize",
    "rmse": "minimize",
    "mae": "minimize",
}


class CVConfig(BaseModel):
    """Cross-validation configuration.

    Attributes:
        strategy: Splitting strategy. ``shuffle_split`` (regression default),
            ``kfold`` (clustering default), or ``stratified_kfold``.
        n_splits: Number of folds / iterations.
        test_size: Held-out fraction (only used by ``shuffle_split``).
        random_state: Seed for reproducibility.
        shuffle: Whether to shuffle before splitting.
    """

    strategy: Literal["shuffle_split", "kfold", "stratified_kfold"] = "shuffle_split"
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True


class OptimizationConfig(BaseModel):
    """Optuna study / optimization configuration.

    Attributes:
        enabled: If False, the runner trains with default params and skips the search.
        n_trials: Number of Optuna trials.
        direction: ``maximize`` (R², Silhouette) or ``minimize`` (MSE, Davies-Bouldin).
        study_name: Optuna study name. Auto-derived from the experiment name if omitted.
        storage: Optuna storage URL (SQLite by default).
        load_if_exists: Resume an existing study instead of erroring.
        timeout: Optional wall-clock timeout in seconds.
        n_jobs: Parallel Optuna workers.
        params: Optional per-experiment param overrides (free-form).
        sampler: Optuna sampler kind.
        pruner: Optuna pruner kind, or ``None`` for no pruning.
    """

    enabled: bool = True
    n_trials: int = 100
    direction: Literal["maximize", "minimize"] = "maximize"
    study_name: str | None = None
    storage: str = "sqlite:///optuna_studies.db"
    load_if_exists: bool = True
    timeout: int | None = None
    n_jobs: int = 1
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sampler: Literal["tpe", "random", "cmaes", "nsgaii"] = "tpe"
    pruner: Literal["median", "percentile", "hyperband", "nop"] | None = None


class ExperimentConfig(BaseModel):
    """Full experiment configuration.

    This is the top-level object produced by :func:`load_config`. It ties
    together a dataset, a model, a cross-validation strategy, and an Optuna
    optimization block.

    Attributes:
        name: Experiment name. Used as default ``study_name`` suffix.
        task: ``regression``, ``clustering`` (M0) or ``classification`` (M4).
        dataset: Dataset name resolvable via :func:`load_dataset_config`.
        model: Model name resolvable via :func:`load_model_config`.
        cv: Cross-validation configuration.
        optimization: Optuna optimization configuration.
        output_dir: Directory for artifacts (predictions, models, plots).
        save_predictions: Persist predictions after the search.
        save_model: Persist the best model after the search.
        random_state: Global seed for the experiment.
        scaler: Optional scaler name (``standard_scaler``, ``minmax_scaler``).
        reducer: Optional reducer name (``pca``, ``umap``).
        metric: Regression optimization metric (M1). When set, drives the
            objective scorer AND defaults ``optimization.direction``
            (``r2``→maximize, others→minimize). None falls back to the
            estimator's default scorer (R² for linear models).
        feature_sets: Named feature subsets for feature-set selection (M1).
            When set, the regression objective samples a feature-set name
            categorically per trial and slices ``X`` to that subset. None
            disables feature-set selection (use all columns).
        models: List of model names for model-selection-as-categorical (M2).
            When set (non-empty), the runner searches across the listed model
            families: each trial samples ``model_type`` categorically and
            builds the chosen model with its own namespaced param_space. Works
            for both regression and clustering. None uses the single ``model``.
    """

    name: str
    task: TaskType
    dataset: str
    model: str
    cv: CVConfig = Field(default_factory=CVConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    output_dir: str = "results"
    save_predictions: bool = False
    save_model: bool = False
    random_state: int = 42
    scaler: str | None = None
    reducer: str | None = None
    metric: RegressionMetric | None = None
    feature_sets: dict[str, list[str]] | None = None
    models: list[str] | None = None

    @model_validator(mode="after")
    def _default_study_name(self) -> ExperimentConfig:
        """Default ``optimization.study_name`` to ``{name}_optuna`` when unset.

        Also defaults the CV strategy to ``kfold`` for clustering tasks when the
        caller did not specify one explicitly, and derives the optimization
        direction from ``metric`` when the metric is set and direction was not
        explicitly pinned.
        """
        if not self.optimization.study_name:
            self.optimization.study_name = f"{self.name}_optuna"
        if self.task == "clustering" and self.cv.strategy == "shuffle_split":
            # Clustering has no target to stratify/shuffle on; KFold over rows
            # is the canonical stability-evaluation strategy.
            self.cv.strategy = "kfold"
        # When a regression metric is set, it dictates the optimization
        # direction (r2 → maximize; mse/rmse/mae → minimize). The metric is the
        # single source of truth for direction; users who want to flip the sign
        # should choose the matching metric instead.
        if self.metric:
            direction = METRIC_DIRECTION[self.metric]
            assert direction in ("maximize", "minimize")
            self.optimization.direction = direction  # type: ignore[assignment]
        return self


class DatasetConfig(BaseModel):
    """Dataset-specific configuration (one entry per dataset in datasets.yaml).

    Attributes:
        name: Registry key used by :func:`load_dataset_config`.
        path: Local file path (relative to ``Settings.data_dir``). Unused when
            ``source`` is ``"kaggle"`` or ``"sklearn"``.
        target_column: Regression/classification target column (None for clustering).
        feature_columns: Optional subset of feature columns to keep.
        source: Where the data comes from — ``"local"`` (a file, default),
            ``"kaggle"`` (Kaggle via ``kagglehub``), or ``"sklearn"`` (a bundled
            sklearn dataset referenced by ``sklearn_name``).
        download: If True (and ``source == "kaggle"``), fetch via ``kagglehub``.
        kaggle_dataset: Kaggle dataset handle (``owner/slug``).
        sklearn_name: Bundled sklearn dataset name (e.g. ``"iris"``, ``"diabetes"``)
            when ``source == "sklearn"``.
        preprocessing: Free-form preprocessing recipe. Recognized keys:
            ``sep``, ``na_values``, ``names``, ``header``, ``drop_columns``,
            ``numeric_conversion``, ``one_hot_encode``, ``dropna``.
    """

    name: str
    path: str | None = None
    target_column: str | None = None
    feature_columns: list[str] | None = None
    source: Literal["local", "kaggle", "sklearn"] = "local"
    download: bool = False
    kaggle_dataset: str | None = None
    sklearn_name: str | None = None
    preprocessing: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Model-specific configuration (one entry per model in models.yaml).

    Attributes:
        name: Registry key.
        type: Kind of entry — ``regression``, ``clustering``, ``reducer``,
            ``scaler``, or ``classification`` (forward-compat).
        default_params: Fixed constructor kwargs used when not optimizing.
        param_space: Per-hyperparameter Optuna search spec. Keys are param
            names; values have a ``type`` of ``float``/``int``/``categorical``
            with ``low``/``high``/``log`` or ``choices``.
    """

    name: str
    type: ModelType
    default_params: dict[str, Any] = Field(default_factory=dict)
    param_space: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Settings(BaseSettings):
    """Global settings loaded from environment variables / ``.env``.

    Environment variables use the ``CO_`` prefix (e.g. ``CO_DATA_DIR``).
    """

    model_config = SettingsConfigDict(
        env_prefix="CO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Root directory used by the YAML loaders. Defaults to the repository root.
    dir: str = str(PROJECT_ROOT)
    optuna_storage: str = "sqlite:///optuna_studies.db"
    clustering_storage: str = "sqlite:///clustering_optuna.db"
    data_dir: str = "data"
    results_dir: str = "results"
    log_level: str = "INFO"


settings = Settings()


def _base_dir(base_dir: str | Path | None = None) -> Path:
    """Resolve the base directory used to locate ``config/`` and ``data/``."""
    return Path(base_dir).expanduser().resolve() if base_dir else Path(settings.dir)


def load_config(config_path: str | Path, base_dir: str | Path | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Args:
        config_path: Path to the experiment YAML. The file may either be a flat
            dict matching :class:`ExperimentConfig` fields, or a dict wrapped
            under a top-level ``experiment:`` key.
        base_dir: Optional base directory override (defaults to :class:`Settings`).

    Returns:
        The validated :class:`ExperimentConfig`.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = _base_dir(base_dir) / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty config file: {path}")
    if "experiment" in data:
        data = data["experiment"]

    return ExperimentConfig(**data)


def load_dataset_config(dataset_name: str, base_dir: str | Path | None = None) -> DatasetConfig:
    """Look up a dataset by name in ``<base_dir>/config/datasets.yaml``.

    Raises:
        FileNotFoundError: If ``datasets.yaml`` is missing.
        ValueError: If ``dataset_name`` is not registered.
    """
    config_path = _base_dir(base_dir) / "config" / "datasets.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)

    for ds in data.get("datasets", []):
        if ds.get("name") == dataset_name:
            return DatasetConfig(**ds)

    raise ValueError(f"Dataset '{dataset_name}' not found in {config_path}")


def load_model_config(model_name: str, base_dir: str | Path | None = None) -> ModelConfig:
    """Look up a model by name in ``<base_dir>/config/models.yaml``.

    Raises:
        FileNotFoundError: If ``models.yaml`` is missing.
        ValueError: If ``model_name`` is not registered.
    """
    config_path = _base_dir(base_dir) / "config" / "models.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)

    for model in data.get("models", []):
        if model.get("name") == model_name:
            return ModelConfig(**model)

    raise ValueError(f"Model '{model_name}' not found in {config_path}")


def get_experiment_configs(base_dir: str | Path | None = None) -> list[ExperimentConfig]:
    """Load every ``*.yaml`` experiment in ``<base_dir>/config/experiments/``.

    Returns an empty list if the directory is missing. Files that fail to parse
    are skipped with a logged warning (never raise).
    """
    configs_dir = _base_dir(base_dir) / "config" / "experiments"
    if not configs_dir.is_dir():
        return []

    configs: list[ExperimentConfig] = []
    for config_file in sorted(configs_dir.glob("*.yaml")):
        try:
            configs.append(load_config(config_file, base_dir=base_dir))
        except Exception:  # noqa: BLE001 — best-effort batch loader
            pass
    return configs


__all__ = [
    "PROJECT_ROOT",
    "ModelType",
    "TaskType",
    "CVConfig",
    "OptimizationConfig",
    "ExperimentConfig",
    "DatasetConfig",
    "ModelConfig",
    "Settings",
    "settings",
    "load_config",
    "load_dataset_config",
    "load_model_config",
    "get_experiment_configs",
]
