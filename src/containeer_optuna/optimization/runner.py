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
    make_clustering_objective,
    make_model_selection_objective,
    make_regression_objective,
)

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

        When ``config.models`` is set (non-empty), dispatch to the
        model-selection objective regardless of task (it handles both
        regression and clustering internally). Otherwise, use the
        single-model objective for the task.
        """
        if self.X is None:
            self.load_data()

        assert self.X is not None  # for mypy

        # Model-selection mode (M2): search across model families.
        if self.config.models:
            if self.config.task == "regression":
                assert self.y is not None, "regression task requires a target column"
            return make_model_selection_objective(self.config, self.X, self.y)

        if self.config.task == "regression":
            assert self.y is not None, "regression task requires a target column"
            return make_regression_objective(self.config, self.X, self.y)
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


__all__ = ["OptunaRunner"]
