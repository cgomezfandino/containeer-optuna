"""Tests for the statistics subpackage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from containeer_optuna.statistics import (
    StatResult,
    anderson_test,
    chi_square,
    correlation_matrix,
    dagostino_test,
    describe,
    kendall_correlation,
    kruskal_wallis,
    mann_whitney_u,
    one_way_anova,
    paired_ttest,
    pearson_correlation,
    shapiro_test,
    spearman_correlation,
    two_sample_ttest,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# --- Hypothesis tests ---------------------------------------------------


def test_two_sample_ttest_returns_stat_result(rng):
    a = rng.normal(0, 1, 100)
    b = rng.normal(0.5, 1, 100)
    r = two_sample_ttest(a, b)
    assert isinstance(r, StatResult)
    assert r.test_name == "Two-sample t-test"
    assert r.pvalue is not None
    assert 0.0 <= r.pvalue <= 1.0


def test_two_sample_ttest_significant(rng):
    a = rng.normal(0, 0.1, 200)
    b = rng.normal(5, 0.1, 200)
    r = two_sample_ttest(a, b)
    assert r.pvalue < 0.001


def test_paired_ttest(rng):
    a = rng.normal(10, 1, 30)
    b = a + rng.normal(0.5, 0.1, 30)
    r = paired_ttest(a, b)
    assert r.test_name == "Paired t-test"
    assert r.pvalue is not None


def test_mann_whitney_u(rng):
    a = rng.exponential(1, 50)
    b = rng.exponential(2, 50)
    r = mann_whitney_u(a, b)
    assert r.test_name == "Mann-Whitney U"
    assert r.pvalue is not None


def test_one_way_anova_significant():
    groups = [
        np.random.RandomState(0).normal(0, 0.1, 50),
        np.random.RandomState(1).normal(5, 0.1, 50),
        np.random.RandomState(2).normal(10, 0.1, 50),
    ]
    r = one_way_anova(*groups)
    assert r.test_name == "One-way ANOVA"
    assert r.pvalue < 0.001
    assert r.extra["df_between"] == 2
    assert r.extra["df_within"] == 147


def test_one_way_anova_requires_two_groups():
    with pytest.raises(ValueError, match="at least 2"):
        one_way_anova([1, 2, 3])


def test_kruskal_wallis(rng):
    groups = [
        rng.normal(0, 1, 30),
        rng.normal(3, 1, 30),
    ]
    r = kruskal_wallis(*groups)
    assert r.test_name == "Kruskal-Wallis"
    assert r.pvalue < 0.05


# --- Normality ----------------------------------------------------------


def test_shapiro_normal_data(rng):
    x = rng.normal(0, 1, 200)
    r = shapiro_test(x)
    assert r.test_name == "Shapiro-Wilk"
    assert r.pvalue > 0.05  # cannot reject normality


def test_shapiro_non_normal_data(rng):
    x = rng.uniform(0, 1, 200)
    r = shapiro_test(x)
    assert r.pvalue < 0.05  # rejects normality


def test_dagostino(rng):
    x = rng.normal(0, 1, 200)
    r = dagostino_test(x)
    assert r.test_name == "D'Agostino K²"
    assert r.pvalue > 0.05


def test_anderson_returns_critical_values(rng):
    x = rng.normal(0, 1, 100)
    r = anderson_test(x)
    assert r.test_name == "Anderson-Darling"
    assert r.pvalue is None  # Anderson doesn't produce a p-value
    assert len(r.extra["critical_values"]) > 0
    assert len(r.extra["significance_levels"]) > 0


# --- Correlation --------------------------------------------------------


def test_pearson_correlation(rng):
    x = rng.normal(0, 1, 100)
    y = 2 * x + rng.normal(0, 0.1, 100)
    r = pearson_correlation(x, y)
    assert r.test_name == "Pearson correlation"
    assert r.statistic > 0.95
    assert r.extra["correlation"] > 0.95
    assert r.pvalue < 0.001


def test_spearman_correlation():
    x = np.arange(1, 51, dtype=float)
    y = x**2  # monotonic but non-linear
    r = spearman_correlation(x, y)
    assert r.statistic > 0.99


def test_kendall_correlation():
    x = np.arange(1, 51, dtype=float)
    y = np.arange(50, 0, -1, dtype=float)  # perfectly anti-monotonic
    r = kendall_correlation(x, y)
    assert r.statistic < -0.99


def test_correlation_matrix():
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(0, 1, 100),
            "c": rng.normal(0, 1, 100),
        }
    )
    corr, pval = correlation_matrix(df, method="pearson")
    assert corr.shape == (3, 3)
    assert pval.shape == (3, 3)
    # Diagonal should be 1.0.
    assert abs(corr.iloc[0, 0] - 1.0) < 1e-6


def test_correlation_matrix_unknown_method():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(ValueError, match="Unknown method"):
        correlation_matrix(df, method="nope")


# --- Categorical --------------------------------------------------------


def test_chi_square_independent():
    # Independent variables → high p-value.
    table = np.array([[50, 50], [50, 50]])
    r = chi_square(table)
    assert r.test_name == "Chi-square"
    assert r.pvalue > 0.05
    assert r.extra["df"] == 1


def test_chi_square_associated():
    # Strongly associated → low p-value.
    table = np.array([[90, 10], [10, 90]])
    r = chi_square(table)
    assert r.pvalue < 0.001


# --- Descriptive --------------------------------------------------------


def test_describe():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    d = describe(data)
    assert d["n"] == 5
    assert d["mean"] == 3.0
    assert d["median"] == 3.0
    assert d["min"] == 1.0
    assert d["max"] == 5.0
    assert d["q1"] == 2.0
    assert d["q3"] == 4.0


def test_describe_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        describe([])
