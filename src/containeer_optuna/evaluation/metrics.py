"""Evaluation metrics for containeer-optuna.

Wraps sklearn metrics with a uniform, typed interface so objectives can call a
single function regardless of the task.

For clustering, :func:`clustering_metrics` returns all three canonical metrics
(Silhouette, Calinski-Harabasz, Davies-Bouldin) plus the number of clusters
found. The convention throughout the framework is:

* **Silhouette** is the *primary* optimization metric (maximize). Range [-1, 1].
* **Calinski-Harabasz** (variance ratio) is stored as a ``user_attr`` (maximize).
* **Davies-Bouldin** (within/between cluster ratio) is stored as a
  ``user_attr`` (minimize; lower is better).

For regression, :func:`regression_metrics` returns R², MSE, RMSE, MAE. The
primary metric is pluggable via the ``metric`` field on ``ExperimentConfig``
(M1): choose ``r2`` (maximize, default), or ``mse``/``rmse``/``mae`` (minimize).
Use :func:`get_regression_scorer` to resolve a metric name into a sklearn
scorer + the matching optimization direction.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    get_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)

ArrayLike = Union[np.ndarray, "list"]

# --- Pluggable regression scorers (M1) -----------------------------------
#
# Maps a friendly metric name → (sklearn scorer name, optimization direction).
# The sklearn scorer name encodes the sign so that higher is always better
# inside ``cross_validate``; the direction tells Optuna how to set up the study.
REGRESSION_SCORERS: dict[str, tuple[str, str]] = {
    "r2": ("r2", "maximize"),
    "mse": ("neg_mean_squared_error", "minimize"),
    "rmse": ("neg_root_mean_squared_error", "minimize"),
    "mae": ("neg_mean_absolute_error", "minimize"),
}


def get_regression_scorer(metric: str) -> tuple[object, str]:
    """Return ``(scorer, direction)`` for a regression metric name.

    Args:
        metric: One of ``"r2"``, ``"mse"``, ``"rmse"``, ``"mae"``.

    Returns:
        A tuple ``(scorer, direction)`` where ``scorer`` is a sklearn scorer
        callable (sign baked in so higher is always better inside
        ``cross_validate``) and ``direction`` is ``"maximize"`` or
        ``"minimize"`` for the Optuna study.

    Raises:
        ValueError: If ``metric`` is not a recognized regression metric.
    """
    if metric not in REGRESSION_SCORERS:
        raise ValueError(
            f"Unknown regression metric '{metric}'. Known metrics: {sorted(REGRESSION_SCORERS)}"
        )
    scorer_name, direction = REGRESSION_SCORERS[metric]
    return get_scorer(scorer_name), direction


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground-truth targets.
        y_pred: Predicted targets.

    Returns:
        Dict with keys ``r2``, ``mse``, ``mae``, ``rmse``.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    return {
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
    }


def clustering_metrics(X: ArrayLike, labels: ArrayLike) -> dict[str, float]:
    """Compute standard (internal) clustering metrics.

    Silhouette and Calinski-Harabasz are *higher-is-better*; Davies-Bouldin is
    *lower-is-better*. Silhouette is the default Optuna objective metric.

    Args:
        X: Feature matrix (n_samples, n_features).
        labels: Predicted cluster labels per sample. ``-1`` labels (DBSCAN
            noise) **must be stripped by the caller** before calling this —
            the objective helper handles that.

    Returns:
        Dict with keys ``silhouette``, ``calinski_harabasz``,
        ``davies_bouldin``, ``n_clusters``.

    Raises:
        ValueError: If fewer than 2 distinct labels are present (clustering
            metrics are undefined in that case).
    """
    X_arr = np.asarray(X)
    labels_arr = np.asarray(labels)
    unique = set(labels_arr.tolist())
    if len(unique) < 2:
        raise ValueError(f"Need at least 2 distinct cluster labels to score; got {sorted(unique)}")

    return {
        "silhouette": float(silhouette_score(X_arr, labels_arr)),
        "calinski_harabasz": float(calinski_harabasz_score(X_arr, labels_arr)),
        "davies_bouldin": float(davies_bouldin_score(X_arr, labels_arr)),
        "n_clusters": float(len(unique)),
    }


__all__ = [
    "REGRESSION_SCORERS",
    "get_regression_scorer",
    "regression_metrics",
    "clustering_metrics",
]
