"""Regression-gated evaluation for reasoning checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .reasoning import ReasoningCase, ReasoningReport
from .regression import RegressionReport, compare_scores
from .reasoning_checkpoint import evaluate_reasoning_checkpoint


@dataclass(frozen=True)
class ReasoningRegressionReport:
    """Current reasoning evaluation plus its baseline regression gate."""

    baseline: ReasoningReport
    current: ReasoningReport
    regression: RegressionReport

    @property
    def passed(self) -> bool:
        return self.regression.passed


def compare_reasoning_checkpoints(
    baseline_checkpoint: Path,
    current_checkpoint: Path,
    cases: Iterable[ReasoningCase],
    predict: Callable,
    *,
    threshold: float = 0.0,
    device: str = "cpu",
) -> ReasoningRegressionReport:
    """Evaluate two checkpoints on the same cases and reject unacceptable regressions."""
    case_list = tuple(cases)
    if not case_list:
        raise ValueError("reasoning regression requires at least one case")
    baseline = evaluate_reasoning_checkpoint(
        baseline_checkpoint, case_list, predict, device=device, name="reasoning-baseline"
    )
    current = evaluate_reasoning_checkpoint(
        current_checkpoint, case_list, predict, device=device, name="reasoning-current"
    )
    regression = compare_scores(baseline.score, current.score, threshold)
    return ReasoningRegressionReport(baseline, current, regression)
