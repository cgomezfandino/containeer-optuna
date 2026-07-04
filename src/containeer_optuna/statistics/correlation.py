"""Correlation tests: Pearson, Spearman, Kendall, and correlation matrix.

All functions wrap ``scipy.stats`` with a uniform :class:`~containeer_optuna.statistics.types.StatResult`
return type.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .types import StatResult

ArrayLike = Any  # np.ndarray | list | pd.Series


def pearson_correlation(x: ArrayLike, y: ArrayLike) -> StatResult:
    """Pearson correlation coefficient (linear correlation).

    Tests whether two samples are linearly correlated. Assumes both variables
    are normally distributed (use :func:`spearman_correlation` otherwise).

    Args:
        x: First variable (1-D).
        y: Second variable (1-D, same length as ``x``).

    Returns:
        :class:`StatResult` with ``statistic=r``, ``pvalue``, and
        ``extra={"correlation": r}`` (alias for convenience).
    """
    r, p = stats.pearsonr(x, y)
    return StatResult(
        test_name="Pearson correlation",
        statistic=float(r),
        pvalue=float(p),
        extra={"correlation": float(r)},
    )


def spearman_correlation(x: ArrayLike, y: ArrayLike) -> StatResult:
    """Spearman rank-order correlation (monotonic, non-parametric).

    Does NOT assume linearity or normality. Detects any monotonic relationship.

    Args:
        x: First variable.
        y: Second variable.

    Returns:
        :class:`StatResult` with ``statistic=rho``, ``pvalue``, and
        ``extra={"correlation": rho}``.
    """
    result = stats.spearmanr(x, y)
    return StatResult(
        test_name="Spearman correlation",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra={"correlation": float(result.statistic)},
    )


def kendall_correlation(x: ArrayLike, y: ArrayLike) -> StatResult:
    """Kendall's tau rank correlation (ordinal association).

    More robust to outliers than Pearson/Spearman; preferred for small samples
    with ties.

    Args:
        x: First variable.
        y: Second variable.

    Returns:
        :class:`StatResult` with ``statistic=tau``, ``pvalue``, and
        ``extra={"correlation": tau}``.
    """
    result = stats.kendalltau(x, y)
    return StatResult(
        test_name="Kendall tau",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra={"correlation": float(result.statistic)},
    )


def correlation_matrix(
    df: pd.DataFrame, method: str = "pearson"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the full correlation matrix + p-value matrix.

    Args:
        df: A DataFrame with numeric columns.
        method: ``"pearson"``, ``"spearman"``, or ``"kendall"``.

    Returns:
        A tuple ``(corr_matrix, pval_matrix)`` of two DataFrames, both
        indexed by column names. The p-value matrix is symmetric; the diagonal
        is 0.
    """
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(cols)
    corr_data = np.zeros((n, n))
    pval_data = np.zeros((n, n))

    func = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }.get(method)
    if func is None:
        raise ValueError(f"Unknown method '{method}'. Use pearson/spearman/kendall.")

    for i in range(n):
        for j in range(i, n):
            if i == j:
                corr_data[i, j] = 1.0
                pval_data[i, j] = 0.0
            else:
                result = func(df[cols[i]], df[cols[j]])
                corr_data[i, j] = corr_data[j, i] = float(result[0])
                pval_data[i, j] = pval_data[j, i] = float(result[1])

    return (
        pd.DataFrame(corr_data, index=cols, columns=cols),
        pd.DataFrame(pval_data, index=cols, columns=cols),
    )


__all__ = [
    "pearson_correlation",
    "spearman_correlation",
    "kendall_correlation",
    "correlation_matrix",
]
