"""Tests for pipeline assembly."""

from __future__ import annotations

import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from containeer_optuna.pipelines import BasePipeline, get_pipeline


def test_get_pipeline_model_only():
    p = get_pipeline("ridge")
    assert isinstance(p, Pipeline)
    assert len(p.steps) == 1
    assert isinstance(p.steps[-1][1], Ridge)
    assert p.steps[-1][0] == BasePipeline.FINAL_STEP


def test_get_pipeline_scaler_model():
    p = get_pipeline("ridge", scaler="standard_scaler")
    assert len(p.steps) == 2
    assert isinstance(p.steps[0][1], StandardScaler)


def test_get_pipeline_scaler_reducer_model():
    p = get_pipeline("kmeans", scaler="standard_scaler", reducer="pca")
    assert len(p.steps) == 3
    assert isinstance(p.steps[0][1], StandardScaler)
    assert isinstance(p.steps[1][1], PCA)
    assert isinstance(p.steps[2][1], KMeans)


def test_get_pipeline_fit_predict(small_clustering_data):
    X = small_clustering_data
    p = get_pipeline("kmeans", scaler="standard_scaler")
    labels = p.fit_predict(X)
    assert labels.shape[0] == X.shape[0]


def test_get_pipeline_unknown_model_raises():
    with pytest.raises(KeyError):
        get_pipeline("not_a_model")


# --- M3: new reducer step types ---------------------------------------


def test_get_pipeline_with_truncated_svd_reducer():
    from sklearn.decomposition import TruncatedSVD

    p = get_pipeline("kmeans", scaler="standard_scaler", reducer="truncated_svd")
    assert isinstance(p.steps[1][1], TruncatedSVD)


def test_get_pipeline_with_factor_analysis_reducer():
    from sklearn.decomposition import FactorAnalysis

    p = get_pipeline("kmeans", scaler="standard_scaler", reducer="factor_analysis")
    assert isinstance(p.steps[1][1], FactorAnalysis)


def test_tsne_in_pipeline_raises_at_fit(small_clustering_data):
    """t-SNE has no transform(), so sklearn Pipeline.fit_predict raises.

    This is a known sklearn limitation: Pipeline._validate_steps() requires
    all intermediate steps to implement transform. t-SNE only implements
    fit_transform. Use plot_best_embedding (which applies steps manually) to
    visualize t-SNE embeddings — not a YAML reducer.
    """
    p = get_pipeline("kmeans", scaler="standard_scaler", reducer="tsne")
    from sklearn.manifold import TSNE

    assert isinstance(p.steps[1][1], TSNE)
    with pytest.raises(TypeError, match="transform"):
        p.fit_predict(small_clustering_data)
