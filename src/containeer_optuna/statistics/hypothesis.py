"""Hypothesis tests: t-tests, Mann-Whitney U, ANOVA, Kruskal-Wallis.

All functions wrap ``scipy.stats`` with a uniform :class:`~containeer_optuna.statistics.types.StatResult`
return type. They are standalone utilities — not Optuna objectives.
"""

from __future__ import annotations

from typing import Any

from scipy import stats

from .types import StatResult

ArrayLike = Any  # np.ndarray | list | pd.Series


def two_sample_ttest(a: ArrayLike, b: ArrayLike) -> StatResult:
    """Independent two-sample t-test (Student or Welch via ``equal_var``).

    Tests whether the means of two independent samples are significantly
    different. Assumes normality of each sample (use :func:`mann_whitney_u`
    if that assumption is violated).

    Args:
        a: First sample (1-D).
        b: Second sample (1-D).

    Returns:
        :class:`StatResult` with ``statistic=t``, ``pvalue``, and
        ``extra={"df": degrees_of_freedom}``.
    """
    result = stats.ttest_ind(a, b)
    return StatResult(
        test_name="Two-sample t-test",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra={"df": float(result.df)} if hasattr(result, "df") else {},
    )


def paired_ttest(a: ArrayLike, b: ArrayLike) -> StatResult:
    """Paired (related) two-sample t-test.

    Tests whether the mean difference between paired observations is zero.

    Args:
        a: First sample (pre-treatment, etc.).
        b: Second sample (post-treatment, etc.). Must have the same length as ``a``.

    Returns:
        :class:`StatResult` with ``statistic=t``, ``pvalue``, and
        ``extra={"df": n-1}``.
    """
    result = stats.ttest_rel(a, b)
    return StatResult(
        test_name="Paired t-test",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra={"df": float(len(a) - 1)},
    )


def mann_whitney_u(a: ArrayLike, b: ArrayLike) -> StatResult:
    """Mann-Whitney U rank test (non-parametric alternative to the two-sample t-test).

    Tests whether one sample is stochastically greater than the other. Does NOT
    assume normality.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        :class:`StatResult` with ``statistic=U``, ``pvalue``.
    """
    result = stats.mannwhitneyu(a, b)
    return StatResult(
        test_name="Mann-Whitney U",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra={},
    )


def one_way_anova(*groups: ArrayLike) -> StatResult:
    """One-way ANOVA (parametric test for equal means across ≥2 groups).

    Tests whether the means of two or more independent groups are equal.
    Assumes normality and homoscedasticity (use :func:`kruskal_wallis` if
    those assumptions are violated).

    Args:
        *groups: Two or more arrays of sample values, one per group.

    Returns:
        :class:`StatResult` with ``statistic=F``, ``pvalue``, and
        ``extra={"df_between": k-1, "df_within": N-k}``.
    """
    if len(groups) < 2:
        raise ValueError("one_way_anova requires at least 2 groups")
    result = stats.f_oneway(*groups)
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    return StatResult(
        test_name="One-way ANOVA",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra={"df_between": k - 1, "df_within": n_total - k},
    )


def kruskal_wallis(*groups: ArrayLike) -> StatResult:
    """Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA).

    Tests whether ≥2 independent samples originate from the same distribution.
    Does NOT assume normality.

    Args:
        *groups: Two or more arrays of sample values.

    Returns:
        :class:`StatResult` with ``statistic=H``, ``pvalue``.
    """
    if len(groups) < 2:
        raise ValueError("kruskal_wallis requires at least 2 groups")
    result = stats.kruskal(*groups)
    return StatResult(
        test_name="Kruskal-Wallis",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra={},
    )


__all__ = [
    "two_sample_ttest",
    "paired_ttest",
    "mann_whitney_u",
    "one_way_anova",
    "kruskal_wallis",
]
