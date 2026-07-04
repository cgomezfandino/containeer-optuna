"""Objective functions for Optuna studies.

Two objective factories are provided:

* :func:`make_regression_objective` — builds a pipeline per trial, runs
  cross-validation with the model's default scorer (R²), and returns the mean
  CV score. Mirrors the regression notebooks' pattern.

* :func:`make_clustering_objective` — builds a pipeline per trial, evaluates
  cluster stability across KFold splits, and returns the mean Silhouette while
  storing Calinski-Harabasz and Davies-Bouldin as ``trial.user_attrs``.

The clustering objective fixes two latent issues present in the original
``perplexity.ipynb``:

1. The notebook re-fit the clusterer directly on the test fold, bypassing the
   scaler/reducer steps of the pipeline. Here we always score via the full
   pipeline, ensuring the scaler/reducer are applied consistently.
2. The notebook did not strip DBSCAN noise (label ``-1``) before computing the
   metrics, which corrupts Silhouette/CH/DB. Here noise points are removed
   before scoring (and folds with fewer than 2 surviving clusters are skipped).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, cross_validate

from ...config import CVConfig, ExperimentConfig
from ...evaluation.metrics import clustering_metrics
from ...pipelines import get_pipeline


@runtime_checkable
class CVSplitter(Protocol):
    """Structural type for sklearn CV splitters (KFold, ShuffleSplit, ...)."""

    def split(self, X: Any, y: Any = ..., groups: Any = ...) -> Any: ...


def make_cv_splitter(cv: CVConfig) -> CVSplitter:
    """Build a sklearn CV splitter from a :class:`CVConfig`.

    Returns one of :class:`ShuffleSplit`, :class:`KFold`, or
    :class:`StratifiedKFold`.
    """
    if cv.strategy == "shuffle_split":
        return ShuffleSplit(  # type: ignore[no-any-return]
            n_splits=cv.n_splits,
            test_size=cv.test_size,
            random_state=cv.random_state,
        )
    if cv.strategy == "kfold":
        return KFold(  # type: ignore[no-any-return]
            n_splits=cv.n_splits, shuffle=cv.shuffle, random_state=cv.random_state
        )
    if cv.strategy == "stratified_kfold":
        return StratifiedKFold(  # type: ignore[no-any-return]
            n_splits=cv.n_splits, shuffle=cv.shuffle, random_state=cv.random_state
        )
    raise ValueError(f"Unknown CV strategy: {cv.strategy}")


def make_regression_objective(
    config: ExperimentConfig,
    X: Any,
    y: Any,
) -> Callable[[Any], float]:
    """Build a regression Optuna objective (mean CV R²).

    Args:
        config: The experiment config (drives pipeline assembly + CV).
        X: Feature matrix.
        y: Target vector.

    Returns:
        A function ``objective(trial) -> float`` returning mean CV R².
    """
    splitter = make_cv_splitter(config.cv)

    def objective(trial: Any) -> float:
        pipeline = get_pipeline(
            model=config.model,
            scaler=config.scaler,
            reducer=config.reducer,
            trial=trial,
            namespace=config.model,
            random_state=config.random_state,
        )
        cv_results = cross_validate(pipeline, X, y, cv=splitter)
        mean_r2 = float(np.mean(cv_results["test_score"]))

        # Store secondary metrics as user_attrs (visible in the dashboard).
        trial.set_user_attr("mean_r2", mean_r2)
        return mean_r2

    return objective


def make_clustering_objective(
    config: ExperimentConfig,
    X: np.ndarray,
) -> Callable[[Any], float]:
    """Build a clustering Optuna objective (mean CV Silhouette).

    Each trial:

    1. Builds the full ``scaler? -> reducer? -> clusterer`` pipeline.
    2. Iterates KFold splits. On each fold, fits the pipeline on the train
       split and produces labels on the test split *through the full pipeline*
       (so scaler/reducer are applied consistently — fixes the notebook bug).
    3. Strips DBSCAN noise points (label ``-1``) before computing metrics.
    4. Skips folds with fewer than 2 distinct labels (metrics are undefined).
    5. Returns mean Silhouette across surviving folds; stores mean
       Calinski-Harabasz and Davies-Bouldin as ``trial.user_attrs``.

    If no fold yields a valid score, the trial returns ``-1.0`` (a sentinel
    worse than any valid Silhouette, which is in [-1, 1]).

    Args:
        config: The experiment config.
        X: Feature matrix (no target for clustering).

    Returns:
        A function ``objective(trial) -> float``.
    """
    splitter = make_cv_splitter(config.cv)
    X_arr = np.asarray(X)

    def objective(trial: Any) -> float:
        silhouettes, chs, dbs = [], [], []

        for train_idx, test_idx in splitter.split(X_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]

            pipeline = get_pipeline(
                model=config.model,
                scaler=config.scaler,
                reducer=config.reducer,
                trial=trial,
                namespace=config.model,
                random_state=config.random_state,
            )

            try:
                pipeline.fit(X_train)
                labels = pipeline.predict(X_test)
            except Exception:
                # Some param combos are degenerate (e.g. GMM singular covariance).
                return -1.0

            # Strip DBSCAN noise: metrics treat -1 as a cluster label, which is wrong.
            non_noise = labels != -1
            if non_noise.sum() < 2:
                continue
            labels_clean = labels[non_noise]
            X_test_clean = X_test[non_noise]

            if len(set(labels_clean.tolist())) < 2:
                continue

            try:
                m = clustering_metrics(X_test_clean, labels_clean)
            except Exception:
                continue

            silhouettes.append(m["silhouette"])
            chs.append(m["calinski_harabasz"])
            dbs.append(m["davies_bouldin"])

        if not silhouettes:
            return -1.0

        trial.set_user_attr("mean_calinski_harabasz", float(np.mean(chs)))
        trial.set_user_attr("mean_davies_bouldin", float(np.mean(dbs)))
        trial.set_user_attr("n_valid_folds", len(silhouettes))
        return float(np.mean(silhouettes))

    return objective


__all__ = [
    "make_cv_splitter",
    "make_regression_objective",
    "make_clustering_objective",
]
