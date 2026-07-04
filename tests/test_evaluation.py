"""Tests for the evaluation subpackage (metrics + model cards)."""

from __future__ import annotations

import numpy as np
import pytest

from containeer_optuna.evaluation import (
    all_model_cards,
    card_to_dict,
    clustering_metrics,
    get_model_card,
    regression_metrics,
)


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    m = regression_metrics(y_true, y_pred)
    assert set(m) == {"r2", "mse", "mae", "rmse"}
    assert 0.0 <= m["r2"] <= 1.0
    assert m["mse"] >= 0
    assert m["rmse"] == pytest.approx(np.sqrt(m["mse"]))


def test_clustering_metrics_on_blobs():
    rng = np.random.RandomState(0)
    a = rng.normal(0, 0.05, size=(20, 2))
    b = rng.normal([5, 5], 0.05, size=(20, 2))
    X = np.vstack([a, b])
    labels = np.array([0] * 20 + [1] * 20)
    m = clustering_metrics(X, labels)
    assert m["silhouette"] > 0.8
    assert m["n_clusters"] == 2


def test_clustering_metrics_single_cluster_raises():
    with pytest.raises(ValueError, match="at least 2"):
        clustering_metrics(np.array([[0.0], [1.0]]), np.array([0, 0]))


def test_all_model_cards_present():
    cards = {c.name for c in all_model_cards()}
    # Every M0 model must have a card.
    for m in [
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
    ]:
        assert m in cards


def test_get_model_card_unknown_returns_none():
    assert get_model_card("nope") is None


def test_card_to_dict_serializable():
    card = get_model_card("ridge")
    d = card_to_dict(card)
    assert d["name"] == "ridge"
    assert isinstance(d["pros"], list)
