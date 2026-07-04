"""Tests for the optimization subpackage (OptunaRunner + objectives)."""

from __future__ import annotations

import numpy as np
import pytest

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


# --- M1: pluggable metrics, feature-set selection, new models ----------


def test_regression_with_mse_metric(small_regression_data):
    """metric='mse' must set direction to minimize and return a negative score."""
    X, y = small_regression_data
    cfg = _in_memory_cfg(
        name="reg_mse",
        task="regression",
        dataset="diabetes",
        model="ridge",
        scaler="standard_scaler",
        metric="mse",
        optimization=OptimizationConfig(n_trials=2),
    )
    cfg.cv = CVConfig(strategy="shuffle_split", n_splits=2)
    assert cfg.optimization.direction == "minimize"  # derived from metric
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = np.asarray(y)
    study = runner.run(n_trials=2, show_progress_bar=False)
    assert len(study.trials) == 2
    # MSE scorer returns negative values (sign-adjusted); best_value ≤ 0.
    assert study.best_value <= 0


def test_regression_feature_set_selection(small_regression_data):
    """feature_sets must be sampled per trial and stored as a user_attr."""
    import pandas as pd

    X_np, y = small_regression_data
    # Build a DataFrame so feature_sets column slicing works.
    X = pd.DataFrame(X_np, columns=["a", "b", "c"])
    cfg = _in_memory_cfg(
        name="reg_fs",
        task="regression",
        dataset="diabetes",
        model="ridge",
        scaler="standard_scaler",
        metric="r2",
        feature_sets={"pair_ab": ["a", "b"], "pair_bc": ["b", "c"], "all": ["a", "b", "c"]},
        optimization=OptimizationConfig(n_trials=3),
    )
    cfg.cv = CVConfig(strategy="shuffle_split", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = X  # DataFrame, not numpy
    runner.y = np.asarray(y)
    study = runner.run(n_trials=3, show_progress_bar=False)
    assert len(study.trials) == 3
    # Every trial must have sampled a feature_set.
    for t in study.trials:
        assert "feature_set" in t.user_attrs
        assert t.user_attrs["feature_set"] in {"pair_ab", "pair_bc", "all"}
    # best_params must include the feature_set choice.
    assert "feature_set" in study.best_params


def test_regression_random_forest_e2e(small_regression_data):
    """A tree ensemble (random_forest) must run end-to-end on synthetic data."""
    X, y = small_regression_data
    cfg = _in_memory_cfg(
        name="reg_rf",
        task="regression",
        dataset="diabetes",
        model="random_forest",
        scaler="standard_scaler",
        metric="r2",
        optimization=OptimizationConfig(n_trials=2),
    )
    cfg.cv = CVConfig(strategy="shuffle_split", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = np.asarray(y)
    study = runner.run(n_trials=2, show_progress_bar=False)
    assert len(study.trials) == 2
    # RF params should appear namespaced.
    assert any(k.startswith("random_forest_") for k in study.best_params)


# --- M2: clustering fix, new models, model-selection --------------------


def test_dbscan_fix_produces_valid_scores(small_clustering_data):
    """REGRESSION: DBSCAN used to always score -1.0 because the objective called
    pipeline.predict(), which DBSCAN doesn't implement. After the fit_predict
    rewrite, DBSCAN must produce valid Silhouette scores on separable blobs."""
    X = small_clustering_data
    cfg = _in_memory_cfg(
        name="dbscan_fix",
        task="clustering",
        dataset="iris",
        model="dbscan",
        scaler="standard_scaler",
        optimization=OptimizationConfig(n_trials=4, direction="maximize"),
    )
    cfg.cv = CVConfig(strategy="kfold", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = None
    study = runner.run(n_trials=4, show_progress_bar=False)
    vals = [t.value for t in study.trials if t.value is not None]
    # At least one trial must NOT be the -1.0 sentinel (the fix works).
    assert any(v != -1.0 for v in vals), f"DBSCAN still broken: all values={vals}"


@pytest.mark.parametrize("model", ["kmeans", "agglomerative", "spectral", "birch", "optics"])
def test_m2_clustering_models_e2e(small_clustering_data, model):
    """Each new clustering model must run end-to-end and produce a study."""
    X = small_clustering_data
    cfg = _in_memory_cfg(
        name=f"clu_{model}",
        task="clustering",
        dataset="iris",
        model=model,
        scaler="standard_scaler",
        optimization=OptimizationConfig(n_trials=2, direction="maximize"),
    )
    cfg.cv = CVConfig(strategy="kfold", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = None
    study = runner.run(n_trials=2, show_progress_bar=False)
    assert len(study.trials) == 2


def test_clustering_model_selection_e2e(small_clustering_data):
    """Model-selection mode: search across clustering families."""
    X = small_clustering_data
    cfg = _in_memory_cfg(
        name="clu_ms",
        task="clustering",
        dataset="iris",
        model="kmeans",  # fallback
        models=["kmeans", "agglomerative", "birch"],
        scaler="standard_scaler",
        optimization=OptimizationConfig(n_trials=4, direction="maximize"),
    )
    cfg.cv = CVConfig(strategy="kfold", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = None
    study = runner.run(n_trials=4, show_progress_bar=False)
    assert len(study.trials) == 4
    # Every trial samples model_type; best_params includes it.
    sampled = {t.params.get("model_type") for t in study.trials}
    assert sampled.issubset({"kmeans", "agglomerative", "birch"})
    assert len(sampled) >= 1
    assert "model_type" in study.best_params


def test_regression_model_selection_e2e(small_regression_data):
    """Model-selection also works for regression (M1 models)."""
    X, y = small_regression_data
    cfg = _in_memory_cfg(
        name="reg_ms",
        task="regression",
        dataset="diabetes",
        model="ridge",
        models=["ridge", "random_forest"],
        scaler="standard_scaler",
        metric="r2",
        optimization=OptimizationConfig(n_trials=2, direction="maximize"),
    )
    cfg.cv = CVConfig(strategy="shuffle_split", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = np.asarray(y)
    study = runner.run(n_trials=2, show_progress_bar=False)
    assert len(study.trials) == 2
    assert "model_type" in study.best_params


# --- M3: reducers + visualization -------------------------------------


def test_clustering_with_truncated_svd_reducer(small_clustering_data):
    """TruncatedSVD (transform-capable) must work as a reducer in clustering."""
    X = small_clustering_data
    cfg = _in_memory_cfg(
        name="clu_svd",
        task="clustering",
        dataset="iris",
        model="kmeans",
        scaler="standard_scaler",
        reducer="truncated_svd",
        optimization=OptimizationConfig(n_trials=2, direction="maximize"),
    )
    cfg.cv = CVConfig(strategy="kfold", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = None
    study = runner.run(n_trials=2, show_progress_bar=False)
    assert len(study.trials) == 2
    # SVD params namespaced.
    assert any(k.startswith("truncated_svd_") for k in study.best_params)


def test_plot_best_embedding_returns_figure(small_clustering_data):
    """runner.plot_best_embedding() must produce a matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")

    X = small_clustering_data
    cfg = _in_memory_cfg(
        name="clu_plot",
        task="clustering",
        dataset="iris",
        model="kmeans",
        scaler="standard_scaler",
        reducer="pca",
        optimization=OptimizationConfig(n_trials=2, direction="maximize"),
    )
    cfg.cv = CVConfig(strategy="kfold", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = None
    runner.run(n_trials=2, show_progress_bar=False)
    fig = runner.plot_best_embedding()
    assert type(fig).__name__ == "Figure"


def test_plot_best_embedding_requires_clustering(small_regression_data):
    """plot_best_embedding must refuse regression tasks with a clear error."""
    X, y = small_regression_data
    cfg = _in_memory_cfg(
        name="reg_plot",
        task="regression",
        dataset="diabetes",
        model="ridge",
        scaler="standard_scaler",
        reducer="pca",
        metric="r2",
        optimization=OptimizationConfig(n_trials=1),
    )
    cfg.cv = CVConfig(strategy="shuffle_split", n_splits=2)
    runner = OptunaRunner(cfg)
    runner.X = np.asarray(X)
    runner.y = np.asarray(y)
    runner.run(n_trials=1, show_progress_bar=False)
    with pytest.raises(ValueError, match="clustering"):
        runner.plot_best_embedding()
