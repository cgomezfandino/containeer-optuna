"""Tests for the data subpackage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from containeer_optuna.data import get_dataset
from containeer_optuna.data.datasets import _preprocess


def test_iris_loads_without_download():
    """The bundled Iris fallback (no path, no kaggle) must load."""
    ds = get_dataset("iris")
    loaded = ds.load()
    # Clustering dataset → returns just X
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape[0] == 150
    assert loaded.shape[1] >= 4


def test_preprocess_drops_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]})
    out = _preprocess(df, {"drop_columns": ["b"]})
    assert list(out.columns) == ["a", "c"]


def test_preprocess_numeric_conversion():
    df = pd.DataFrame({"hp": ["100", "?", "120"]})
    out = _preprocess(df, {"numeric_conversion": ["hp"]})
    # "?" becomes NaN
    assert pd.isna(out.loc[1, "hp"])
    assert out.loc[0, "hp"] == 100.0


def test_preprocess_dropna():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    out = _preprocess(df, {"dropna": True})
    assert len(out) == 2


def test_preprocess_one_hot_encode():
    df = pd.DataFrame({"cat": ["x", "y", "x"], "num": [1, 2, 3]})
    out = _preprocess(df, {"one_hot_encode": ["cat"]})
    assert "cat_x" in out.columns
    assert "cat_y" in out.columns


def test_auto_mpg_preprocessing_recipe():
    """The Auto MPG '?' handling must be encapsulated in the recipe."""
    cfg = get_dataset("auto_mpg").config
    assert cfg.preprocessing["na_values"] == ["?"]
    assert "horsepower" in cfg.preprocessing["numeric_conversion"]


def test_get_dataset_unknown_raises():
    with pytest.raises(ValueError):
        get_dataset("totally_made_up")


# --- M1: bundled diabetes dataset --------------------------------------


def test_diabetes_loads_with_target():
    """Bundled diabetes dataset (source: sklearn) — no download, returns (X, y)."""
    X, y = get_dataset("diabetes").load()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (442, 10)
    assert y.shape == (442,)
    # sklearn diabetes feature names
    assert list(X.columns) == ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def test_diabetes_config_has_source_sklearn():
    cfg = get_dataset("diabetes").config
    assert cfg.source == "sklearn"
    assert cfg.sklearn_name == "diabetes"
    assert cfg.target_column == "target"


def test_iris_still_loads_after_refactor():
    """The iris dataset now uses source: sklearn; ensure backward compat.

    Iris is a clustering dataset (no target_column), so the loader returns the
    full frame (4 features + the species column), 150 rows.
    """
    X = get_dataset("iris").load()
    assert X.shape[0] == 150
    # The 4 human-readable feature names must be present.
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        assert col in X.columns
