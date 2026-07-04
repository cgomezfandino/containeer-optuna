"""Tests for the NLP subpackage (M8).

All tests are guarded by ``pytest.importorskip("transformers")`` — they skip
if transformers is not installed (e.g. on CI without the ``[nlp]`` extra).
"""

from __future__ import annotations

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from containeer_optuna import (  # noqa: E402
    CVConfig,
    ExperimentConfig,
    OptimizationConfig,
    OptunaRunner,
    all_model_cards,
    get_model_card,
)


def test_transformer_module_builds():
    """build_transformer_module must produce a working model."""
    from containeer_optuna.models.dl.transformer import build_transformer_module

    model = build_transformer_module("distilbert-base-uncased", num_labels=2)
    # Check it has the expected architecture.
    assert hasattr(model, "classifier") or hasattr(model, "pre_classifier")


def test_tokenizer_works():
    """build_tokenizer must tokenize text correctly."""
    from containeer_optuna.models.dl.transformer import build_tokenizer

    tokenizer = build_tokenizer("distilbert-base-uncased", max_seq_length=32)
    encoded = tokenizer(
        ["hello world", "goodbye"],
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    assert "input_ids" in encoded
    assert encoded["input_ids"].shape[1] == 32


def test_nlp_objective_e2e():
    """NLP transformer fine-tuning must run end-to-end."""
    # Synthetic text data (small for speed).
    texts = np.array(
        [
            "I love this product, it is amazing!",
            "This is terrible, I hate it.",
            "Absolutely wonderful experience.",
            "Worst purchase ever, do not buy.",
            "Great quality and fast delivery.",
            "Disappointing and overpriced.",
            "Highly recommend to everyone.",
            "Complete waste of money.",
        ]
        * 5
    )  # 40 samples
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 5)

    cfg = ExperimentConfig(
        name="nlp_e2e",
        task="classification",
        dataset="ag_news",
        model="transformer_classifier",
        metric="accuracy",
        cv=CVConfig(strategy="stratified_kfold", n_splits=2),
        optimization=OptimizationConfig(n_trials=2, direction="maximize"),
    )
    cfg.optimization.storage = "sqlite:///:memory:"
    cfg.optimization.load_if_exists = False
    cfg.optimization.pruner = "median"

    runner = OptunaRunner(cfg)
    runner.X = texts
    runner.y = y
    study = runner.run(n_trials=2, show_progress_bar=False)
    assert len(study.trials) == 2
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    assert len(completed) >= 1


def test_nlp_model_card_present():
    cards = {c.name for c in all_model_cards()}
    assert "transformer_classifier" in cards


def test_nlp_card_marked_m8():
    assert get_model_card("transformer_classifier").milestone == "M8"
