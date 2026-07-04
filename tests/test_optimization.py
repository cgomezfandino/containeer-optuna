"""Tests for the optimization subpackage (OptunaRunner + objectives)."""

from __future__ import annotations

import numpy as np

from containeer_optuna.config import CVConfig, ExperimentConfig, OptimizationConfig
from containeer_optuna.optimization import (
    OptunaRunner,
    make_clustering_objective,
    make_cv_splitter,
    make_regression_objective,
)


def _in_memory_cfg(**kw) -> ExperimentConfig:
    cfg = ExperimentConfig(**kw)
    cfg.optimization.storage = "sqlite:///:memory:"
    cfg.optimization.load_if_exists = False
    return cfg


def test_make_cv_splitter_strategies():
    from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold

    assert isinstance(make_cv_splitter(CVConfig(strategy="shuffle_split")), ShuffleSplit)
    assert isinstance(make_cv_splitter(CVConfig(strategy="kfold")), KFold)
    assert isinstance(make_cv_splitter(CVConfig(strategy="stratified_kfold")), StratifiedKFold)


def test_regression_objective_returns_float(small_regression_data):
    X, y = small_regression_data
    cfg = _in_memory_cfg(
        name="reg",
        task="regression",
        dataset="auto_mpg",
        model="ridge",
        scaler="standard_scaler",
    )
    obj = make_regression_objective(cfg, X, y)

    class FakeTrial:
        def __init__(self):
            self.user_attrs = {}

        def set_user_attr(self, k, v):
            self.user_attrs[k] = v

        def suggest_float(self, *a, **k):
            return 1.0

        def suggest_int(self, *a, **k):
            return 1

        def suggest_categorical(self, *a, **k):
            return None

    val = obj(FakeTrial())
    assert isinstance(val, float)
    assert -1.0 <= val <= 1.0


def test_clustering_objective_on_blobs(small_clustering_data):
    X = small_clustering_data
    cfg = _in_memory_cfg(
        name="clu",
        task="clustering",
        dataset="iris",
        model="kmeans",
        scaler="standard_scaler",
    )
    obj = make_clustering_objective(cfg, X)

    class FakeTrial:
        def __init__(self):
            self.user_attrs = {}

        def set_user_attr(self, k, v):
            self.user_attrs[k] = v

        def suggest_int(self, name, low, high, log=False):
            return 3 if "n_clusters" in name else 2

        def suggest_categorical(self, name, choices):
            return choices[0]

        def suggest_float(self, *a, **k):
            return 0.5

    val = obj(FakeTrial())
    # Well-separated blobs → silhouette should be high (> 0.5)
    assert val > 0.5


def test_optuna_runner_end_to_end_regression(small_regression_data):
    X, y = small_regression_data
    cfg = _in_memory_cfg(
        name="reg_e2e",
        task="regression",
        dataset="auto_mpg",
        model="ridge",
        scaler="standard_scaler",
        optimization=OptimizationConfig(n_trials=3, direction="maximize"),
    )
    cfg.cv = CVConfig(strategy="shuffle_split", n_splits=2)
    runner = OptunaRunner(cfg)
    # Inject data directly to avoid needing the auto_mpg file.
    runner.X = np.asarray(X)
    runner.y = np.asarray(y)
    study = runner.run(n_trials=3, show_progress_bar=False)
    assert len(study.trials) == 3
    summary = runner.quick_summary()
    assert "best_value" in summary
    assert "ridge_alpha" in summary["best_params"] or len(summary["best_params"]) > 0


def test_optuna_runner_end_to_end_clustering(small_clustering_data):
    X = small_clustering_data
    cfg = _in_memory_cfg(
        name="clu_e2e",
        task="clustering",
        dataset="iris",
        model="kmeans",
        scaler="standard_scaler",
        optimization=OptimizationConfig(n_trials=3, direction="maximize"),
    )
    cfg.cv = CVConfig(strategy="kfold", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = None
    study = runner.run(n_trials=3, show_progress_bar=False)
    assert len(study.trials) == 3
