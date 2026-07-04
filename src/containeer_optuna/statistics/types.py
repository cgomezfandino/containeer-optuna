"""Common types for the statistics subpackage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StatResult:
    """Uniform return type for all statistical tests.

    Attributes:
        test_name: Human-readable test name (e.g. ``"Shapiro-Wilk"``,
            ``"One-way ANOVA"``).
        statistic: The test statistic (e.g. t, F, W, chi2).
        pvalue: The p-value, or ``None`` if the test doesn't produce one
            (e.g. Anderson-Darling).
        extra: Dict of test-specific extras (e.g. ``{"df": 2}`` for ANOVA
            degrees of freedom, ``{"correlation": 0.85}`` for Pearson r,
            ``{"critical_values": [...]}`` for Anderson-Darling).
    """

    test_name: str
    statistic: float
    pvalue: float | None
    extra: dict[str, Any]


__all__ = ["StatResult"]
