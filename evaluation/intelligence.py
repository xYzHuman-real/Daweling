"""Capability-level intelligence evaluation for Daweling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .metrics import mean_score


@dataclass(frozen=True)
class IntelligenceCase:
    """One capability probe with a bounded scoring function."""
    id: str
    capability: str
    prompt: str
    scorer: Callable[[str], float]
    weight: float = 1.0


@dataclass(frozen=True)
class IntelligenceResult:
    case_id: str
    capability: str
    score: float
    prediction: str


@dataclass(frozen=True)
class IntelligenceReport:
    results: tuple[IntelligenceResult, ...]
    capability_scores: dict[str, float]
    overall_score: float

    @property
    def passed(self) -> bool:
        return bool(self.results) and self.overall_score >= 0.7


def run_intelligence_evaluation(
    model: Callable[[str], str],
    cases: Iterable[IntelligenceCase],
) -> IntelligenceReport:
    """Evaluate reasoning-related capabilities without exposing private reasoning traces."""
    results: list[IntelligenceResult] = []
    weighted: list[tuple[float, float]] = []
    by_capability: dict[str, list[float]] = {}
    for case in cases:
        if case.weight <= 0:
            raise ValueError("case weight must be greater than zero")
        prediction = model(case.prompt)
        score = max(0.0, min(1.0, float(case.scorer(prediction))))
        results.append(IntelligenceResult(case.id, case.capability, score, prediction))
        by_capability.setdefault(case.capability, []).append(score)
        weighted.append((score, case.weight))
    capability_scores = {
        capability: mean_score(scores) for capability, scores in by_capability.items()
    }
    total_weight = sum(weight for _, weight in weighted)
    overall = sum(score * weight for score, weight in weighted) / total_weight if total_weight else 0.0
    return IntelligenceReport(tuple(results), capability_scores, overall)


def contains_all(*terms: str) -> Callable[[str], float]:
    """Create a simple deterministic scorer for required concepts."""
    normalized = tuple(term.casefold() for term in terms if term.strip())
    if not normalized:
        raise ValueError("at least one term is required")

    def score(prediction: str) -> float:
        text = prediction.casefold()
        return sum(term in text for term in normalized) / len(normalized)

    return score
