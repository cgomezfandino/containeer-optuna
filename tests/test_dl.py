"""Tests for the deep learning subpackage (M6).

All tests are guarded by ``pytest.importorskip("torch")`` — they are skipped
entirely if PyTorch is not installed (e.g. on CI without the ``[dl]`` extra).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from containeer_optuna import (  # noqa: E402
    CVConfig,
    ExperimentConfig,
    OptimizationConfig,
    OptunaRunner,
    all_model_cards,
    get_model_card,
)


def test_mlp_module_forward_pass():
    """The MLP nn.Module must produce correctly-shaped output."""
    from containeer_optuna.models.dl.mlp import build_mlp_module

    model = build_mlp_module(input_dim=10, hidden_sizes=[64, 32], output_dim=1, task="regression")
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5, 1)


def test_mlp_module_classification_output():
    from containeer_optuna.models.dl.mlp import build_mlp_module

    model = build_mlp_module(input_dim=10, hidden_sizes=[32], output_dim=3, task="classification")
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 3)  # logits for 3 classes


def test_dl_regression_e2e():
    """MLP regression must run end-to-end with the DL objective."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=80, n_features=6, random_state=42)

    cfg = ExperimentConfig(
        name="dl_reg",
        task="regression",
        dataset="diabetes",
        model="mlp_regressor",
        scaler="standard_scaler",
        metric="r2",
        cv=CVConfig(strategy="shuffle_split", n_splits=2, test_size=0.2),
        optimization=OptimizationConfig(n_trials=2, direction="maximize"),
    )
    cfg.optimization.storage = "sqlite:///:memory:"
    cfg.optimization.load_if_exists = False
    cfg.optimization.pruner = "median"

    runner = OptunaRunner(cfg)
    runner.X = X
    runner.y = y
    study = runner.run(n_trials=2, show_progress_bar=False)
    assert len(study.trials) == 2
    # At least one trial must complete (not all pruned).
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    assert len(completed) >= 1


def test_dl_classification_e2e():
    """MLP classification must run end-to-end."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=80, n_features=6, n_classes=2, random_state=42)

    cfg = ExperimentConfig(
        name="dl_clf",
        task="classification",
        dataset="breast_cancer",
        model="mlp_classifier",
        scaler="standard_scaler",
        metric="accuracy",
        cv=CVConfig(strategy="stratified_kfold", n_splits=2),
        optimization=OptimizationConfig(n_trials=2, direction="maximize"),
    )
    cfg.optimization.storage = "sqlite:///:memory:"
    cfg.optimization.load_if_exists = False
    cfg.optimization.pruner = "median"

    runner = OptunaRunner(cfg)
    runner.X = X
    runner.y = y
    study = runner.run(n_trials=2, show_progress_bar=False)
    assert len(study.trials) == 2
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    assert len(completed) >= 1


def test_dl_epoch_pruning_can_fire():
    """With enough trials and a median pruner, at least one trial should be pruned."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=60, n_features=4, random_state=42)

    cfg = ExperimentConfig(
        name="dl_prune",
        task="regression",
        dataset="diabetes",
        model="mlp_regressor",
        scaler="standard_scaler",
        metric="r2",
        cv=CVConfig(strategy="shuffle_split", n_splits=2, test_size=0.2),
        optimization=OptimizationConfig(n_trials=10, direction="maximize"),
    )
    cfg.optimization.storage = "sqlite:///:memory:"
    cfg.optimization.load_if_exists = False
    cfg.optimization.pruner = "median"

    runner = OptunaRunner(cfg)
    runner.X = X
    runner.y = y
    study = runner.run(n_trials=10, show_progress_bar=False)
    # With median pruning and 10 trials, at least one should be pruned.
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    assert len(pruned) >= 1, f"Expected at least 1 pruned trial; got {len(pruned)}"


def test_mlp_model_cards_present():
    cards = {c.name for c in all_model_cards()}
    assert "mlp_regressor" in cards
    assert "mlp_classifier" in cards


def test_mlp_cards_marked_m6():
    assert get_model_card("mlp_regressor").milestone == "M6"
    assert get_model_card("mlp_classifier").milestone == "M6"
