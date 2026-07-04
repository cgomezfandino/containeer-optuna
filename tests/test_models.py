"""Tests for the model registry and param-space dispatcher."""

from __future__ import annotations

import pytest

from containeer_optuna.models import MODEL_CLASSES, get_model, suggest_params


class _FakeTrial:
    """A deterministic stand-in for optuna.trial.Trial for unit tests."""

    def __init__(self, values):
        self._values = list(values)
        self.calls = []

    def suggest_float(self, name, low, high, log=False):
        self.calls.append(("float", name, low, high, log))
        return self._values.pop(0)

    def suggest_int(self, name, low, high, log=False):
        self.calls.append(("int", name, low, high, log))
        return self._values.pop(0)

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, choices))
        return self._values.pop(0)


def test_all_registered_models_are_known():
    expected = {
        "ridge",
        "lasso",
        "ols",
        "kmeans",
        "dbscan",
        "gmm",
        "pca",
        "umap",
        "standard_scaler",
        "minmax_scaler",
    }
    assert expected.issubset(set(MODEL_CLASSES))


def test_get_model_default_params_ridge():
    est = get_model("ridge")
    assert est.__class__.__name__ == "Ridge"


def test_get_model_default_params_kmeans():
    est = get_model("kmeans")
    assert est.__class__.__name__ == "KMeans"
    assert est.n_clusters == 3  # from models.yaml default


def test_get_model_with_trial_samples_params():
    trial = _FakeTrial([5, "random", "elkan"])  # n_clusters, init, algorithm
    est = get_model("kmeans", trial=trial, namespace="kmeans")
    assert est.n_clusters == 5
    assert est.init == "random"
    assert est.algorithm == "elkan"
    # Confirm namespacing
    called_names = [c[1] for c in trial.calls]
    assert called_names == ["kmeans_n_clusters", "kmeans_init", "kmeans_algorithm"]


def test_get_model_unknown_raises():
    with pytest.raises(KeyError):
        get_model("not_a_model")


def test_suggest_params_float_log():
    trial = _FakeTrial([2.5])
    out = suggest_params(trial, {"alpha": {"type": "float", "low": 0.1, "high": 10.0, "log": True}})
    assert out == {"alpha": 2.5}
    assert trial.calls[0][4] is True  # log=True


def test_suggest_params_int():
    trial = _FakeTrial([7])
    out = suggest_params(trial, {"n": {"type": "int", "low": 2, "high": 10}})
    assert out == {"n": 7}


def test_suggest_params_categorical():
    trial = _FakeTrial(["elkan"])
    out = suggest_params(trial, {"algo": {"type": "categorical", "choices": ["lloyd", "elkan"]}})
    assert out == {"algo": "elkan"}


def test_suggest_params_unknown_type_raises():
    trial = _FakeTrial([])
    with pytest.raises(ValueError, match="Unknown param_space type"):
        suggest_params(trial, {"x": {"type": "string"}})


def test_suggest_params_namespacing():
    trial = _FakeTrial([3])
    suggest_params(trial, {"n_components": {"type": "int", "low": 2, "high": 10}}, namespace="pca")
    assert trial.calls[0][1] == "pca_n_components"


def test_overrides_win_over_sampled():
    # kmeans param_space has n_clusters (int), init (categorical), algorithm (categorical)
    trial = _FakeTrial([5, "random", "elkan"])
    est = get_model("kmeans", trial=trial, namespace="kmeans", random_state=123)
    assert est.random_state == 123
    assert est.n_clusters == 5


# --- M1: new regression models ------------------------------------------


@pytest.mark.parametrize(
    "name,cls_name",
    [
        ("elasticnet", "ElasticNet"),
        ("decision_tree", "DecisionTreeRegressor"),
        ("random_forest", "RandomForestRegressor"),
        ("gradient_boosting", "GradientBoostingRegressor"),
        ("svr", "SVR"),
    ],
)
def test_m1_models_instantiate_with_defaults(name, cls_name):
    est = get_model(name)
    assert type(est).__name__ == cls_name


def test_elasticnet_with_sampled_params():
    # alpha (float), l1_ratio (float)
    trial = _FakeTrial([0.5, 0.3])
    est = get_model("elasticnet", trial=trial, namespace="elasticnet")
    assert est.alpha == 0.5
    assert est.l1_ratio == 0.3
    # Confirm namespacing
    assert trial.calls[0][1] == "elasticnet_alpha"


def test_random_forest_with_sampled_params():
    # n_estimators (int), max_depth (categorical), min_samples_split (int), max_features (categorical)
    trial = _FakeTrial([200, 10, 5, "log2"])
    est = get_model("random_forest", trial=trial, namespace="random_forest")
    assert est.n_estimators == 200
    assert est.max_depth == 10  # int choice passed through
    assert est.min_samples_split == 5
    assert est.max_features == "log2"


def test_max_depth_none_coercion():
    """The sentinel string 'None' must be coerced to Python None for RF/DT."""
    # n_estimators, max_depth="None", min_samples_split, max_features
    trial = _FakeTrial([100, "None", 2, "sqrt"])
    est = get_model("random_forest", trial=trial, namespace="random_forest")
    assert est.max_depth is None  # coerced from "None"


def test_svr_with_sampled_params():
    # C (float log), epsilon (float log), kernel (categorical)
    trial = _FakeTrial([10.0, 0.01, "linear"])
    est = get_model("svr", trial=trial, namespace="svr")
    assert est.C == 10.0
    assert est.epsilon == 0.01
    assert est.kernel == "linear"


def test_gradient_boosting_with_sampled_params():
    # learning_rate (float log), n_estimators (int), max_depth (int), subsample (float)
    trial = _FakeTrial([0.05, 300, 5, 0.8])
    est = get_model("gradient_boosting", trial=trial, namespace="gradient_boosting")
    assert est.learning_rate == 0.05
    assert est.n_estimators == 300
    assert est.max_depth == 5
    assert est.subsample == 0.8


# --- M2: new clustering models -----------------------------------------


@pytest.mark.parametrize(
    "name,cls_name",
    [
        ("agglomerative", "AgglomerativeClustering"),
        ("spectral", "SpectralClustering"),
        ("birch", "Birch"),
        ("optics", "OPTICS"),
    ],
)
def test_m2_clustering_models_instantiate(name, cls_name):
    est = get_model(name)
    assert type(est).__name__ == cls_name


def test_agglomerative_with_sampled_params():
    # n_clusters (int), linkage (categorical)
    trial = _FakeTrial([4, "average"])
    est = get_model("agglomerative", trial=trial, namespace="agglomerative")
    assert est.n_clusters == 4
    assert est.linkage == "average"


def test_birch_with_sampled_params():
    # n_clusters (int), threshold (float)
    trial = _FakeTrial([5, 0.3])
    est = get_model("birch", trial=trial, namespace="birch")
    assert est.n_clusters == 5
    assert est.threshold == 0.3


def test_optics_with_sampled_params():
    # min_samples (int), xi (float)
    trial = _FakeTrial([10, 0.1])
    est = get_model("optics", trial=trial, namespace="optics")
    assert est.min_samples == 10
    assert abs(est.xi - 0.1) < 1e-6


def test_hdbscan_graceful_when_unavailable():
    """If sklearn < 1.3, get_model('hdbscan') raises ImportError, not KeyError."""
    from containeer_optuna.models import MODEL_CLASSES

    if MODEL_CLASSES.get("hdbscan") is None:
        with pytest.raises(ImportError, match="optional dependency"):
            get_model("hdbscan")
    else:
        # sklearn >= 1.3: hdbscan is available; just check it instantiates.
        est = get_model("hdbscan")
        assert type(est).__name__ == "HDBSCAN"
