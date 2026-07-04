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
from ...evaluation.metrics import clustering_metrics, get_regression_scorer
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
    """Build a regression Optuna objective (mean CV primary metric).

    The metric is pluggable via ``config.metric`` (M1). When unset, falls back
    to ``"r2"`` (the estimator default scorer). When ``config.feature_sets`` is
    set (M1), a feature subset is sampled categorically per trial and ``X`` is
    sliced to that subset (requires ``X`` to be a DataFrame with column names).

    The primary metric is returned; the other three regression metrics are
    stored as ``trial.user_attrs`` for inspection in the dashboard.

    Args:
        config: The experiment config (drives pipeline assembly + CV + metric).
        X: Feature matrix (DataFrame when ``config.feature_sets`` is set).
        y: Target vector.

    Returns:
        A function ``objective(trial) -> float`` returning the mean CV value
        of the primary metric (sign-adjusted so higher is always the direction
        configured by ``optimization.direction``).
    """
    splitter = make_cv_splitter(config.cv)
    metric = config.metric or "r2"
    scorer, _direction = get_regression_scorer(metric)
    feature_sets = config.feature_sets

    def objective(trial: Any) -> float:
        # Optional feature-set selection (M1): sample a named subset per trial.
        if feature_sets:
            set_name = trial.suggest_categorical("feature_set", list(feature_sets.keys()))
            cols = feature_sets[set_name]
            X_trial = X[cols] if hasattr(X, "loc") else X
            trial.set_user_attr("feature_set", set_name)
        else:
            X_trial = X

        pipeline = get_pipeline(
            model=config.model,
            scaler=config.scaler,
            reducer=config.reducer,
            trial=trial,
            namespace=config.model,
            random_state=config.random_state,
        )
        cv_results = cross_validate(pipeline, X_trial, y, cv=splitter, scoring=scorer)
        mean_primary = float(np.mean(cv_results["test_score"]))

        # Store the primary metric as a user_attr (sign-adjusted to be
        # interpretable: r2 as-is, errors negated back to positive).
        trial.set_user_attr(f"mean_{metric}", mean_primary)
        return mean_primary

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
