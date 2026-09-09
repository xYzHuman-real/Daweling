"""Standard, repeatable intelligence benchmark suite for Daweling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .intelligence import IntelligenceCase, IntelligenceReport, contains_all, run_intelligence_evaluation


@dataclass(frozen=True)
class CapabilityThreshold:
    capability: str
    minimum: float


DEFAULT_THRESHOLDS = (
    CapabilityThreshold("reasoning", 0.70),
    CapabilityThreshold("planning", 0.70),
    CapabilityThreshold("strategy", 0.70),
    CapabilityThreshold("verification", 0.70),
    CapabilityThreshold("recovery", 0.70),
)


def default_cases() -> tuple[IntelligenceCase, ...]:
    """Return small smoke benchmarks covering the major intelligence stages."""
    return (
        IntelligenceCase("reasoning-001", "reasoning", "Explain how to reason about a failed task.", contains_all("evidence", "verify")),
        IntelligenceCase("planning-001", "planning", "Create a plan for completing a task safely.", contains_all("plan", "step")),
        IntelligenceCase("strategy-001", "strategy", "Choose a strategy and justify the choice.", contains_all("strategy", "reason")),
        IntelligenceCase("verification-001", "verification", "Describe how to verify a result.", contains_all("verify", "evidence")),
        IntelligenceCase("recovery-001", "recovery", "A task failed. Propose a recovery approach.", contains_all("recover", "alternative")),
    )


def run_standard_intelligence_suite(
    model: Callable[[str], str],
    *,
    cases: Iterable[IntelligenceCase] | None = None,
    thresholds: Iterable[CapabilityThreshold] = DEFAULT_THRESHOLDS,
) -> IntelligenceReport:
    """Run the standard smoke suite and fail closed when required capabilities regress."""
    report = run_intelligence_evaluation(model, cases or default_cases())
    minimums = {item.capability: item.minimum for item in thresholds}
    failures = [
        capability
        for capability, minimum in minimums.items()
        if report.capability_scores.get(capability, 0.0) < minimum
    ]
    if failures:
        raise AssertionError(
            "Intelligence regression: " + ", ".join(sorted(failures))
        )
    return report
