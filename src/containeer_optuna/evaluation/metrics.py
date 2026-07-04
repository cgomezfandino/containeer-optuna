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

For regression, :func:`regression_metrics` returns R², MSE, MAE. The convention
is to optimize **R²** (maximize) by default; M1 will make the metric pluggable.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)

ArrayLike = Union[np.ndarray, "list"]


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


__all__ = ["regression_metrics", "clustering_metrics"]
