"""Chi-square test of independence for categorical data."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .types import StatResult

ArrayLike = Any  # np.ndarray | pd.DataFrame


def chi_square(contingency_table: ArrayLike) -> StatResult:
    """Chi-square test of independence on a contingency table.

    Tests whether two categorical variables are independent. The null
    hypothesis is that they are independent; a small p-value (< 0.05) indicates
    a significant association.

    Args:
        contingency_table: A 2-D contingency table (frequencies), e.g. the
            output of ``pd.crosstab(col_a, col_b)``.

    Returns:
        :class:`StatResult` with ``statistic=chi2``, ``pvalue``, and
        ``extra={"df": dof, "expected_freq": expected_array}``.
    """
    table = np.asarray(contingency_table)
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return StatResult(
        test_name="Chi-square",
        statistic=float(chi2),
        pvalue=float(p),
        extra={
            "df": int(dof),
            "expected_freq": expected.tolist(),
        },
    )


__all__ = ["chi_square"]
