"""Normality tests: Shapiro-Wilk, D'Agostino K², Anderson-Darling.

All functions wrap ``scipy.stats`` with a uniform :class:`~containeer_optuna.statistics.types.StatResult`
return type.
"""

from __future__ import annotations

from typing import Any

from scipy import stats

from .types import StatResult

ArrayLike = Any  # np.ndarray | list | pd.Series


def shapiro_test(x: ArrayLike) -> StatResult:
    """Shapiro-Wilk test for normality.

    The null hypothesis is that the sample is drawn from a normal distribution.
    A p-value < alpha (e.g. 0.05) rejects normality.

    Note: for n > 5000 the test may be unreliable; prefer Anderson-Darling.

    Args:
        x: Sample values (1-D).

    Returns:
        :class:`StatResult` with ``statistic=W``, ``pvalue``.
    """
    W, p = stats.shapiro(x)
    return StatResult(
        test_name="Shapiro-Wilk",
        statistic=float(W),
        pvalue=float(p),
        extra={},
    )


def dagostino_test(x: ArrayLike) -> StatResult:
    """D'Agostino K² test for normality (based on skew and kurtosis).

    A good alternative to Shapiro-Wilk for large samples (n > 5000).

    Args:
        x: Sample values (1-D).

    Returns:
        :class:`StatResult` with ``statistic=K2``, ``pvalue``.
    """
    result = stats.normaltest(x)
    return StatResult(
        test_name="D'Agostino K²",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra={},
    )


def anderson_test(x: ArrayLike, dist: str = "norm") -> StatResult:
    """Anderson-Darling test for goodness-of-fit to a distribution.

    Unlike Shapiro/D'Agostino, this returns critical values at fixed
    significance levels rather than a p-value. Compare the statistic to the
    critical value at your desired alpha; if statistic > critical, reject the
    null (data does NOT come from ``dist``).

    Args:
        x: Sample values (1-D).
        dist: Distribution to test against (``"norm"``, ``"expon"``,
            ``"logistic"``, ``"gumbel_l"``, ``"gumbel_r"``).

    Returns:
        :class:`StatResult` with ``statistic=A2``, ``pvalue=None`` (Anderson
        doesn't produce a p-value), and
        ``extra={"critical_values": [...], "significance_levels": [...]}``.
    """
    result = stats.anderson(x, dist=dist)
    return StatResult(
        test_name="Anderson-Darling",
        statistic=float(result.statistic),
        pvalue=None,
        extra={
            "critical_values": [float(v) for v in result.critical_values],
            "significance_levels": [float(v) for v in result.significance_level],
            "dist": dist,
        },
    )


__all__ = ["shapiro_test", "dagostino_test", "anderson_test"]
