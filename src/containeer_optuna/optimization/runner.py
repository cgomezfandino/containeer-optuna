"""The :class:`OptunaRunner` — the central execution entry point.

Given an :class:`~containeer_optuna.config.ExperimentConfig`, the runner:

1. Loads the dataset via :func:`~containeer_optuna.data.get_dataset`.
2. Builds an Optuna objective via :mod:`containeer_optuna.optimization.objectives`.
3. Constructs the study (sampler + pruner + storage + study_name).
4. Runs ``study.optimize(...)`` and returns the resulting :class:`optuna.Study`.

The runner also provides :meth:`OptunaRunner.quick_summary` and a plotting
helper that mirrors the canonical "slice / param-importance / contour" trio
used throughout the notebooks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..data import get_dataset
from ..utils import get_logger, seed_all, study_summary
from .objectives import (
    make_classification_objective,
    make_clustering_objective,
    make_dl_objective,
    make_model_selection_objective,
    make_regression_objective,
)

# Models that use the PyTorch DL objective (custom training loop + pruning).
_DL_MODELS = {"mlp_regressor", "mlp_classifier", "cnn_classifier", "rnn_classifier"}

# Models that use the NLP objective (transformer fine-tuning + pruning).
_NLP_MODELS = {"transformer_classifier"}

try:
    import optuna
    from optuna.pruners import (
        HyperbandPruner,
        MedianPruner,
        NopPruner,
        PercentilePruner,
    )
    from optuna.samplers import RandomSampler, TPESampler
except ImportError as e:  # pragma: no cover — optuna is a hard dependency
    raise ImportError("optuna is required: pip install optuna") from e


# CMAES / NSGAII samplers live in optional submodules of optuna. Import them
# defensively; if unavailable we fall back to TPESampler at construction time.
def _get_cmaes_sampler() -> type | None:
    try:
        from optuna.samplers import CMAESSampler  # type: ignore[attr-defined]

        return CMAESSampler  # type: ignore[no-any-return]
    except ImportError:
        try:
            from optuna.integration.cma import CMAESSampler  # type: ignore[attr-defined]

            return CMAESSampler  # type: ignore[no-any-return]
        except ImportError:
            return None


def _get_nsgaii_sampler() -> type | None:
    try:
        from optuna.samplers.nsgaii import NSGAIISampler  # type: ignore[attr-defined]

        return NSGAIISampler  # type: ignore[no-any-return]
    except ImportError:
        try:
            from optuna.samplers import NSGAIISampler  # type: ignore[attr-defined]

            return NSGAIISampler  # type: ignore[no-any-return]
        except ImportError:
            return None


_SAMPLERS: dict = {
    "tpe": TPESampler,
    "random": RandomSampler,
    "cmaes": _get_cmaes_sampler(),
    "nsgaii": _get_nsgaii_sampler(),
}

_PRUNERS = {
    "median": MedianPruner,
    "percentile": PercentilePruner,
    "hyperband": HyperbandPruner,
    "nop": NopPruner,
}


class OptunaRunner:
    """Run an Optuna study from a :class:`ExperimentConfig`.

    Example:
        >>> from containeer_optuna import load_config, OptunaRunner
        >>> cfg = load_config("config/experiments/clustering_optimization.yaml")
        >>> runner = OptunaRunner(cfg)
        >>> study = runner.run(n_trials=20)
        >>> print(study.best_value, study.best_params)

    Attributes:
        config: The :class:`ExperimentConfig` driving the run.
        logger: A configured :class:`logging.Logger`.
        X / y: Loaded data (``y`` is None for clustering).
        study: The :class:`optuna.Study` after :meth:`run` is called.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        seed: int | None = None,
        base_dir: str | None = None,
    ) -> None:
        self.config = config
        self.base_dir = base_dir
        self.logger = get_logger("containeer_optuna.runner")
        self._seed = seed if seed is not None else config.random_state
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.study: optuna.Study | None = None

    # -- data -------------------------------------------------------------

    def load_data(self) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | None]:
        """Load the dataset and return ``(X, y)`` (``y`` is None for clustering).

        When the experiment defines ``feature_sets``, ``X`` is kept as a pandas
        DataFrame (so the objective can slice columns by name); otherwise it is
        converted to a numpy array.
        """
        dataset = get_dataset(self.config.dataset, base_dir=self.base_dir)
        loaded = dataset.load()

        if isinstance(loaded, tuple):
            X, y = loaded
        else:
            X, y = loaded, None

        # Preserve the DataFrame when feature-set selection is active so the
        # objective can slice by column name; numpy arrays can't be sliced by
        # feature-set names.
        if self.config.feature_sets:
            self.X = X  # keep DataFrame
        else:
            self.X = np.asarray(X)
        self.y = np.asarray(y) if y is not None else None
        return self.X, self.y

    # -- sampler / pruner -------------------------------------------------

    def _build_sampler(self) -> Any:
        """Construct the Optuna sampler from the config.

        Falls back to ``TPESampler`` if the requested optional sampler
        (CMA-ES / NSGA-II) is not installed in the current Optuna build.
        """
        opt_cfg = self.config.optimization
        sampler_cls = _SAMPLERS.get(opt_cfg.sampler) or _SAMPLERS["tpe"]
        if sampler_cls is None:
            sampler_cls = _SAMPLERS["tpe"]
        kwargs: dict[str, Any] = {"seed": self._seed}
        if sampler_cls is TPESampler:
            kwargs["multivariate"] = True
        return sampler_cls(**kwargs)

    def _build_pruner(self) -> Any:
        """Construct the Optuna pruner from the config (NopPruner if None)."""
        opt_cfg = self.config.optimization
        kind = opt_cfg.pruner
        if not kind:
            return NopPruner()
        return _PRUNERS[kind]()

    # -- objective --------------------------------------------------------

    def _build_objective(self) -> Any:
        """Select and build the right objective for the task.

        Dispatch order:
        1. DL models (mlp_regressor / mlp_classifier) → DL objective with
           epoch pruning.
        2. Model-selection mode (``config.models`` set) → model-selection
           objective.
        3. Single-model sklearn objective per task.
        """
        if self.X is None:
            self.load_data()

        assert self.X is not None  # for mypy

        # NLP models (M8): transformer fine-tuning with tokenization + pruning.
        if self.config.model in _NLP_MODELS:
            assert self.y is not None, "NLP classification requires a target column"
            from .objectives.nlp import make_nlp_objective

            return make_nlp_objective(self.config, self.X, self.y)

        # DL models (M6): custom training loop with epoch pruning.
        if self.config.model in _DL_MODELS:
            assert self.y is not None, f"{self.config.task} task requires a target column"
            return make_dl_objective(self.config, self.X, self.y)

        # Model-selection mode (M2): search across model families.
        if self.config.models:
            if self.config.task in ("regression", "classification"):
                assert self.y is not None, f"{self.config.task} task requires a target column"
            return make_model_selection_objective(self.config, self.X, self.y)

        if self.config.task == "regression":
            assert self.y is not None, "regression task requires a target column"
            return make_regression_objective(self.config, self.X, self.y)
        if self.config.task == "classification":
            assert self.y is not None, "classification task requires a target column"
            return make_classification_objective(self.config, self.X, self.y)
        if self.config.task == "clustering":
            return make_clustering_objective(self.config, self.X)
        raise ValueError(f"Unsupported task: {self.config.task}")

    # -- run --------------------------------------------------------------

    def run(
        self,
        n_trials: int | None = None,
        show_progress_bar: bool = True,
    ) -> optuna.Study:
        """Execute the Optuna study and return the resulting :class:`Study`.

        Args:
            n_trials: Override the configured ``optimization.n_trials``.
            show_progress_bar: Whether to show Optuna's progress bar.
        """
        seed_all(self._seed)
        if self.X is None:
            self.load_data()

        opt_cfg = self.config.optimization
        if not opt_cfg.enabled:
            self.logger.warning(
                "Optimization disabled in config — falling back to a 1-trial "
                "default-params evaluation."
            )

        objective = self._build_objective()
        sampler = self._build_sampler()
        pruner = self._build_pruner()

        study = optuna.create_study(
            direction=opt_cfg.direction,
            study_name=opt_cfg.study_name,
            storage=opt_cfg.storage,
            load_if_exists=opt_cfg.load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

        trials = n_trials if n_trials is not None else opt_cfg.n_trials
        self.logger.info(
            "Running study '%s' (%s, n_trials=%d, sampler=%s, pruner=%s)",
            opt_cfg.study_name,
            opt_cfg.direction,
            trials if opt_cfg.enabled else 1,
            opt_cfg.sampler,
            opt_cfg.pruner or "none",
        )

        study.optimize(
            objective,
            n_trials=trials if opt_cfg.enabled else 1,
            timeout=opt_cfg.timeout,
            n_jobs=opt_cfg.n_jobs,
            show_progress_bar=show_progress_bar,
            catch=(Exception,),
        )

        self.study = study
        return study

    # -- reporting --------------------------------------------------------

    def quick_summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the last run."""
        if self.study is None:
            raise RuntimeError("Call .run() before .quick_summary()")
        return study_summary(self.study)

    def plot_standard(self) -> dict[str, object]:
        """Return the canonical Optuna plotly figures for the last study.

        Returns a dict with keys ``optimization_history``, ``param_importances``,
        ``slice``, ``contour`` (each an Optuna FigureWidget-or-Figure).
        """
        if self.study is None:
            raise RuntimeError("Call .run() before .plot_standard()")

        import optuna.visualization as vis

        return {
            "optimization_history": vis.plot_optimization_history(self.study),
            "param_importances": vis.plot_param_importances(self.study),
            "slice": vis.plot_slice(self.study),
            "contour": vis.plot_contour(self.study),
        }

    def plot_best_embedding(self) -> object:
        """Plot a 2D embedding of ``self.X`` using the best trial's reducer.

        Rebuilds the best-trial pipeline (scaler → reducer → clusterer) from
        ``study.best_params``, fits it on ``self.X``, extracts the reducer's
        2D projection, and scatters it colored by the clusterer's labels.

        Requires:
        * The study must have run (``self.study`` not None, ≥1 trial).
        * ``self.X`` must be loaded.
        * The experiment must use a reducer (``config.reducer``) and be a
          clustering task. For regression or no-reducer experiments this raises
          a clear error.

        Returns:
            The matplotlib :class:`Figure` with the 2D scatter.
        """
        if self.study is None or not self.study.trials:
            raise RuntimeError("Call .run() before .plot_best_embedding()")
        if self.X is None:
            raise RuntimeError("No data loaded; call .load_data() first")
        if self.config.task != "clustering":
            raise ValueError(
                "plot_best_embedding is only supported for clustering tasks "
                f"(got task={self.config.task!r})"
            )
        if not self.config.reducer:
            raise ValueError(
                "plot_best_embedding requires a reducer in the experiment config "
                "(set `reducer:` to pca / tsne / umap / truncated_svd / factor_analysis)"
            )

        from ..evaluation.plotting import plot_embedding_2d

        best_params = dict(self.study.best_params)

        # Reconstruct each estimator with the best params (de-namespaced).
        reducer_name = self.config.reducer
        # For model-selection studies, the best model_type wins; else use config.model.
        model_name = best_params.pop("model_type", self.config.model)
        # Feature-set choice (if any) is not relevant for embedding viz.
        best_params.pop("feature_set", None)

        reducer_overrides = self._denamespace(best_params, reducer_name)
        model_overrides = self._denamespace(best_params, model_name)

        from ..models import MODEL_CLASSES, get_model

        # Build the scaler (if any).
        scaler = get_model(self.config.scaler) if self.config.scaler else None
        reducer = get_model(
            reducer_name, random_state=self.config.random_state, **reducer_overrides
        )
        # Filter model_overrides to kwargs the estimator accepts.
        import inspect

        cls = MODEL_CLASSES.get(model_name)
        if cls is not None:
            valid = set(inspect.signature(cls).parameters) - {"self"}
            model_overrides = {k: v for k, v in model_overrides.items() if k in valid}
        model = get_model(model_name, random_state=self.config.random_state, **model_overrides)

        # Apply steps manually (NOT sklearn.Pipeline — it requires `transform`
        # on intermediate steps, which t-SNE lacks). fit_transform each step in
        # sequence; the clusterer uses fit_predict on the reduced data.
        X_arr = np.asarray(self.X)
        if scaler is not None:
            X_arr = scaler.fit_transform(X_arr)
        # Reducer: use fit_transform (works for both t-SNE-no-transform and
        # PCA/SVD/FA/UMAP).
        embedding = reducer.fit_transform(X_arr)
        labels = model.fit_predict(embedding)

        X2d = np.asarray(embedding)[:, :2]
        return plot_embedding_2d(
            X2d,
            labels=labels,
            title=f"Best embedding — {self.config.optimization.study_name}",
        )

    @staticmethod
    def _denamespace(params: dict[str, Any], prefix: str) -> dict[str, Any]:
        """Strip a ``<prefix>_`` namespace from matching keys.

        Keys not starting with the prefix are ignored. E.g. with prefix "tsne",
        ``{"tsne_perplexity": 20, "kmeans_n_clusters": 3}`` → ``{"perplexity": 20}``.
        """
        out: dict[str, Any] = {}
        p = f"{prefix}_"
        for k, v in params.items():
            if k.startswith(p):
                out[k[len(p) :]] = v
        return out


__all__ = ["OptunaRunner"]
