"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def experiments_dir(repo_root: Path) -> Path:
    return repo_root / "config" / "experiments"


@pytest.fixture
def small_regression_data():
    """A tiny deterministic regression dataset."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-3, 3, size=(60, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.1, size=60)
    return X, y


@pytest.fixture
def small_clustering_data():
    """Three well-separated blobs for clustering tests."""
    rng = np.random.RandomState(42)
    a = rng.normal(0, 0.1, size=(30, 2))
    b = rng.normal([5, 5], 0.1, size=(30, 2))
    c = rng.normal([5, 0], 0.1, size=(30, 2))
    return np.vstack([a, b, c])


@pytest.fixture
def small_classification_data():
    """A tiny deterministic binary classification dataset (0/1 targets)."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    return X, y
