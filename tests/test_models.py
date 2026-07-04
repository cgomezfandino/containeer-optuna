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
