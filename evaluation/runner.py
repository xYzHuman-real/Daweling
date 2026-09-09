"""Dataset-driven evaluation runner for simple model interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .metrics import exact_match, mean_score


@dataclass(frozen=True)
class EvaluationExample:
    prompt: str
    reference: str


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    score: float


def evaluate_exact_match(
    model: Callable[[str], str], examples: Iterable[EvaluationExample]
) -> EvaluationReport:
    scores = [exact_match(model(example.prompt), example.reference) for example in examples]
    if not scores:
        return EvaluationReport(total=0, score=0.0)
    return EvaluationReport(total=len(scores), score=mean_score(scores))
