"""Reasoning-focused evaluation primitives for Daweling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .metrics import exact_match, mean_score


@dataclass(frozen=True)
class ReasoningCase:
    """A small reasoning task with an expected final answer."""

    id: str
    prompt: str
    expected: str
    category: str = "general"


@dataclass(frozen=True)
class ReasoningResult:
    case_id: str
    category: str
    score: float
    prediction: str


@dataclass(frozen=True)
class ReasoningReport:
    name: str
    results: tuple[ReasoningResult, ...]
    score: float
    category_scores: dict[str, float]


def run_reasoning_evaluation(
    name: str,
    cases: Iterable[ReasoningCase],
    predict: Callable[[str], str],
) -> ReasoningReport:
    """Run deterministic final-answer scoring and aggregate scores by category."""
    results: list[ReasoningResult] = []
    for case in cases:
        prediction = predict(case.prompt)
        results.append(
            ReasoningResult(
                case_id=case.id,
                category=case.category,
                score=exact_match(prediction, case.expected),
                prediction=prediction,
            )
        )
    if not results:
        raise ValueError("reasoning evaluation requires at least one case")

    category_values: dict[str, list[float]] = {}
    for result in results:
        category_values.setdefault(result.category, []).append(result.score)
    category_scores = {
        category: mean_score(scores) for category, scores in sorted(category_values.items())
    }
    return ReasoningReport(
        name=name,
        results=tuple(results),
        score=mean_score(result.score for result in results),
        category_scores=category_scores,
    )
