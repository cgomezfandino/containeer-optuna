"""Descriptive statistics: mean, std, quartiles, skew, kurtosis."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats

ArrayLike = Any  # np.ndarray | list | pd.Series


def describe(data: ArrayLike) -> dict[str, float]:
    """Compute a rich set of descriptive statistics for a 1-D sample.

    Args:
        data: A 1-D array of numeric values.

    Returns:
        A dict with keys: ``n``, ``mean``, ``std``, ``min``, ``q1`` (25th
        percentile), ``median``, ``q3`` (75th percentile), ``max``, ``skew``,
        ``kurtosis`` (excess).
    """
    arr = np.asarray(data, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("describe requires a non-empty array")

    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    skew = float(sp_stats.skew(arr))
    kurtosis = float(sp_stats.kurtosis(arr))  # excess kurtosis (Fisher)

    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(np.max(arr)),
        "skew": skew,
        "kurtosis": kurtosis,
    }


__all__ = ["describe"]
