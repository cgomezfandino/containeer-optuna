"""Statistics: hypothesis tests, normality, correlation, and descriptive stats.

All functions wrap ``scipy.stats`` with a uniform :class:`StatResult` return
type. These are standalone utilities — not Optuna objectives.

Public API:
    StatResult
    # hypothesis
    two_sample_ttest, paired_ttest, mann_whitney_u, one_way_anova,
    kruskal_wallis
    # normality
    shapiro_test, dagostino_test, anderson_test
    # correlation
    pearson_correlation, spearman_correlation, kendall_correlation,
    correlation_matrix
    # categorical
    chi_square
    # descriptive
    describe
"""

from .categorical import chi_square
from .correlation import (
    correlation_matrix,
    kendall_correlation,
    pearson_correlation,
    spearman_correlation,
)
from .descriptive import describe
from .hypothesis import (
    kruskal_wallis,
    mann_whitney_u,
    one_way_anova,
    paired_ttest,
    two_sample_ttest,
)
from .normality import anderson_test, dagostino_test, shapiro_test
from .types import StatResult

__all__ = [
    "StatResult",
    # hypothesis
    "two_sample_ttest",
    "paired_ttest",
    "mann_whitney_u",
    "one_way_anova",
    "kruskal_wallis",
    # normality
    "shapiro_test",
    "dagostino_test",
    "anderson_test",
    # correlation
    "pearson_correlation",
    "spearman_correlation",
    "kendall_correlation",
    "correlation_matrix",
    # categorical
    "chi_square",
    # descriptive
    "describe",
]
