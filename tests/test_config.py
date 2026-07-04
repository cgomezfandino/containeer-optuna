"""Tests for the config system (loaders, validators, bug regressions)."""

from __future__ import annotations

from pathlib import Path

import pytest

from containeer_optuna.config import (
    CVConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    _base_dir,
    load_config,
    load_dataset_config,
    load_model_config,
)


def test_base_dir_resolves_to_repo_root(repo_root: Path):
    """Regression: loaders used to look at src/config instead of <repo>/config."""
    assert _base_dir() == repo_root


def test_load_dataset_config_auto_mpg():
    cfg = load_dataset_config("auto_mpg")
    assert cfg.name == "auto_mpg"
    assert cfg.target_column == "horsepower"
    assert "?" in cfg.preprocessing["na_values"]


def test_load_dataset_config_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        load_dataset_config("does_not_exist")


def test_load_model_config_accepts_reducer_and_scaler():
    """Regression: ModelConfig.type Literal used to reject reducer/scaler."""
    for name, kind in [
        ("pca", "reducer"),
        ("umap", "reducer"),
        ("standard_scaler", "scaler"),
        ("minmax_scaler", "scaler"),
    ]:
        cfg = load_model_config(name)
        assert cfg.type == kind


def test_load_model_config_kmeans():
    cfg = load_model_config("kmeans")
    assert cfg.type == "clustering"
    assert "n_clusters" in cfg.param_space
    assert cfg.param_space["n_clusters"]["type"] == "int"


def test_load_model_config_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        load_model_config("nope")


def test_study_name_defaults_to_experiment_name_suffix(repo_root: Path):
    cfg = ExperimentConfig(
        name="my_exp",
        task="regression",
        dataset="auto_mpg",
        model="ridge",
        optimization=OptimizationConfig(),
    )
    assert cfg.optimization.study_name == "my_exp_optuna"


def test_study_name_explicit_is_respected():
    cfg = ExperimentConfig(
        name="my_exp",
        task="regression",
        dataset="auto_mpg",
        model="ridge",
        optimization=OptimizationConfig(study_name="custom_name"),
    )
    assert cfg.optimization.study_name == "custom_name"


def test_clustering_defaults_to_kfold_cv():
    """Regression: clustering must default to KFold (ShuffleSplit is meaningless
    for unsupervised stability evaluation)."""
    cfg = ExperimentConfig(
        name="c",
        task="clustering",
        dataset="iris",
        model="kmeans",
        optimization=OptimizationConfig(),
    )
    assert cfg.cv.strategy == "kfold"


def test_load_experiment_yaml_clustering(experiments_dir: Path):
    cfg = load_config(experiments_dir / "clustering_optimization.yaml")
    assert cfg.task == "clustering"
    assert cfg.model == "kmeans"
    assert cfg.cv.strategy == "kfold"


def test_load_experiment_yaml_regression(experiments_dir: Path):
    cfg = load_config(experiments_dir / "auto_mpg_model_selection.yaml")
    assert cfg.task == "regression"
    assert cfg.model == "ridge"


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_cvconfig_defaults():
    cv = CVConfig()
    assert cv.strategy == "shuffle_split"
    assert cv.n_splits == 5
    assert cv.random_state == 42


def test_model_config_accepts_classification_type():
    """Forward-compat: classification will land in M4."""
    m = ModelConfig(name="logreg", type="classification")
    assert m.type == "classification"
