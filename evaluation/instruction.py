"""Instruction-following evaluation helpers for Daweling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .metrics import exact_match, mean_score


@dataclass(frozen=True)
class InstructionCase:
    """A prompt with the response expected from the model."""

    id: str
    instruction: str
    expected: str


@dataclass(frozen=True)
class InstructionResult:
    case_id: str
    score: float
    prediction: str


@dataclass(frozen=True)
class InstructionReport:
    name: str
    results: tuple[InstructionResult, ...]
    score: float


def run_instruction_evaluation(
    name: str,
    cases: Iterable[InstructionCase],
    predict: Callable[[str], str],
) -> InstructionReport:
    """Evaluate predictions deterministically using normalized exact match."""
    results_list: list[InstructionResult] = []
    for case in cases:
        prediction = predict(case.instruction)
        results_list.append(
            InstructionResult(
                case_id=case.id,
                score=exact_match(prediction, case.expected),
                prediction=prediction,
            )
        )
    results = tuple(results_list)
    if not results:
        raise ValueError("instruction evaluation requires at least one case")
    return InstructionReport(name=name, results=results, score=mean_score(result.score for result in results))
