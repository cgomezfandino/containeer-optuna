"""Dataset loading and preprocessing for containeer-optuna.

This module provides :class:`BaseDataset`, a YAML-driven :class:`DatasetLoader`,
and a :func:`get_dataset` factory. It supports both local files (e.g. the UCI
Auto MPG dataset with its whitespace-separated, ``?``-for-missing format) and
Kaggle datasets via ``kagglehub``.

The preprocessing recipe is read from :class:`~containeer_optuna.config.DatasetConfig`
and applied in a fixed, sensible order:

1. Read with the configured separator / column names / NA tokens.
2. Drop explicitly-listed columns.
3. Coerce listed columns to numeric (turning unparsable values into NaN).
4. One-hot encode listed categorical columns.
5. Drop rows with any remaining NaN (if ``dropna`` is set).
6. Slice features / target.

For clustering tasks (no ``target_column``), :meth:`BaseDataset.load` returns
only the feature matrix ``X``; for supervised tasks it returns ``(X, y)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import DatasetConfig, settings


class BaseDataset(ABC):
    """Abstract base class for datasets.

    Subclasses must implement :meth:`load`. The concrete
    :class:`YamlDatasetLoader` handles all built-in datasets defined in
    ``config/datasets.yaml``.
    """

    config: DatasetConfig

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    @abstractmethod
    def load(
        self,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess the dataset.

        Returns:
            Either ``X`` (DataFrame, unsupervised) or ``(X, y)`` (supervised).
        """
        raise NotImplementedError


def _read_local(path: Path, preprocessing: dict[str, Any]) -> pd.DataFrame:
    """Read a local data file honoring the preprocessing recipe.

    Args:
        path: Path to the data file.
        preprocessing: Recipe dict. Recognized keys: ``sep``, ``names``,
            ``header``, ``na_values``.

    Returns:
        The raw (un-preprocessed) DataFrame.
    """
    sep = preprocessing.get("sep")
    if sep:
        # YAML may store the separator escaped (e.g. "\\s+", "\\t").
        sep = sep.encode().decode("unicode_escape")
    names = preprocessing.get("names")
    header = preprocessing.get("header")
    na_values = preprocessing.get("na_values")

    read_kwargs: dict[str, Any] = {}
    if names is not None:
        read_kwargs["names"] = names
    if header is not None:
        # ``header=None`` means "no header row"; pandas expects the Python None.
        read_kwargs["header"] = None if header != 0 else 0
    if na_values is not None:
        read_kwargs["na_values"] = na_values

    # Use read_csv for fixed/separated formats. ``sep`` may be a regex (e.g.
    # ``\s+``), which pandas supports via the ``engine="python"`` fallback.
    if sep and (sep in (r"\s+",) or any(ch in sep for ch in "+*?|")):
        read_kwargs["sep"] = sep
        read_kwargs["engine"] = "python"
    elif sep:
        read_kwargs["sep"] = sep

    return pd.read_csv(path, **read_kwargs)


def _preprocess(df: pd.DataFrame, preprocessing: dict[str, Any]) -> pd.DataFrame:
    """Apply the preprocessing recipe to a raw DataFrame.

    The order is: drop columns → numeric coercion → one-hot encode → dropna.
    """
    drop_columns = preprocessing.get("drop_columns") or []
    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    numeric_conversion = preprocessing.get("numeric_conversion") or []
    for col in numeric_conversion:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    one_hot = preprocessing.get("one_hot_encode") or []
    if one_hot:
        encode_cols = [c for c in one_hot if c in df.columns]
        if encode_cols:
            df = pd.get_dummies(df, columns=encode_cols, drop_first=False, dtype=int)

    if preprocessing.get("dropna"):
        df = df.dropna().reset_index(drop=True)

    return df


def _resolve_kaggle_path(kaggle_dataset: str) -> tuple[Path, pd.DataFrame]:
    """Download a Kaggle dataset via ``kagglehub`` and load its first CSV.

    Args:
        kaggle_dataset: Kaggle handle, e.g. ``"owner/slug"``.

    Returns:
        Tuple of (directory path, DataFrame of the first CSV found).
    """
    import kagglehub  # imported lazily — kagglehub is an optional dep at runtime

    dataset_path = Path(kagglehub.dataset_download(kaggle_dataset))
    csv_files = sorted(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in Kaggle dataset '{kaggle_dataset}' at {dataset_path}"
        )
    return dataset_path, pd.read_csv(csv_files[0])


class YamlDatasetLoader(BaseDataset):
    """Concrete dataset loader driven by a :class:`DatasetConfig`.

    Handles four cases: local files (e.g. Auto MPG), Kaggle downloads
    (customer_personality, credit_card), the bundled Iris dataset (loaded via
    sklearn when no path/URL is given), and any custom file path.
    """

    def load(
        self,
        data_dir: str | Path | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """Load, preprocess, and (optionally) split features/target.

        Args:
            data_dir: Optional override for the data root directory. Defaults
                to :attr:`~containeer_optuna.config.Settings.data_dir`.

        Returns:
            ``X`` (DataFrame) for clustering datasets, or ``(X, y)`` for
            supervised datasets that define a ``target_column``.
        """
        cfg = self.config
        preprocessing = cfg.preprocessing or {}

        if cfg.download and cfg.kaggle_dataset:
            _, df = _resolve_kaggle_path(cfg.kaggle_dataset)
            # Kaggle files are plain CSV; honor ``sep`` if provided.
            # (Re-read is unnecessary — we already loaded via pd.read_csv above.)
            df = _preprocess(df, preprocessing)
        elif cfg.name == "iris" and not cfg.path:
            # Bundled fallback: load Iris from sklearn (no download required).
            from sklearn.datasets import load_iris

            bunch = load_iris(as_frame=True)
            df = bunch.frame
            df.columns = [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
                "target",
            ]
            df = _preprocess(df, preprocessing)
        else:
            if not cfg.path:
                raise ValueError(f"Dataset '{cfg.name}' has no path and is not a Kaggle download")
            base = Path(data_dir) if data_dir else Path(settings.data_dir)
            path = Path(cfg.path)
            if not path.is_absolute():
                # Try the path as-is first, then relative to the data dir.
                path = path if path.exists() else base / path
            df = _read_local(path, preprocessing)
            df = _preprocess(df, preprocessing)

        # Optionally restrict to a feature subset.
        if cfg.feature_columns:
            keep = [c for c in cfg.feature_columns if c in df.columns]
            feature_df = df[keep].copy()
        else:
            feature_df = df.copy()

        # Split target if configured.
        if cfg.target_column and cfg.target_column in df.columns:
            y = df[cfg.target_column]
            # Ensure the target is not also in the features.
            if cfg.target_column in feature_df.columns:
                feature_df = feature_df.drop(columns=[cfg.target_column])
            return feature_df, y

        return feature_df


# --- Registry / factory ---------------------------------------------------

# Map of dataset name -> loader class. Currently a single loader handles all
# built-in datasets, but the registry pattern leaves room for special-purpose
# loaders in future milestones.
_DATASET_REGISTRY: dict[str, type] = {
    "auto_mpg": YamlDatasetLoader,
    "customer_personality": YamlDatasetLoader,
    "credit_card": YamlDatasetLoader,
    "iris": YamlDatasetLoader,
}


def get_dataset(
    name: str,
    base_dir: str | Path | None = None,
    **kwargs: Any,
) -> BaseDataset:
    """Build a :class:`BaseDataset` by name.

    The dataset's :class:`DatasetConfig` is looked up in
    ``<base_dir>/config/datasets.yaml``. Unknown names raise ``ValueError``.

    Args:
        name: Dataset registry key (e.g. ``"auto_mpg"``, ``"iris"``).
        base_dir: Optional base directory override.
        **kwargs: Forwarded to the loader constructor (rarely needed).

    Returns:
        A :class:`BaseDataset` ready to :meth:`~BaseDataset.load`.
    """
    from ..config import load_dataset_config

    config = load_dataset_config(name, base_dir=base_dir)
    loader_cls = _DATASET_REGISTRY.get(name, YamlDatasetLoader)
    instance: BaseDataset = loader_cls(config, **kwargs)
    return instance


__all__ = [
    "BaseDataset",
    "YamlDatasetLoader",
    "get_dataset",
]
