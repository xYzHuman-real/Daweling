"""Regression checks for tracking model evaluation against a baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegressionReport:
    baseline: float
    current: float
    delta: float
    threshold: float

    @property
    def passed(self) -> bool:
        return self.delta >= -self.threshold


def compare_scores(baseline: float, current: float, threshold: float = 0.0) -> RegressionReport:
    """Compare a current score with a baseline and allow bounded regression."""
    if not 0.0 <= baseline <= 1.0:
        raise ValueError("baseline must be between 0 and 1")
    if not 0.0 <= current <= 1.0:
        raise ValueError("current must be between 0 and 1")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    return RegressionReport(
        baseline=baseline,
        current=current,
        delta=current - baseline,
        threshold=threshold,
    )
