"""Small benchmark definitions for tracking Daweling regressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .metrics import exact_match, mean_score


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    prompt: str
    expected: str


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    score: float
    prediction: str


@dataclass(frozen=True)
class BenchmarkReport:
    name: str
    results: tuple[BenchmarkResult, ...]
    score: float


def run_benchmark(
    name: str, model: Callable[[str], str], cases: Iterable[BenchmarkCase]
) -> BenchmarkReport:
    results = tuple(
        BenchmarkResult(
            case_id=case.id,
            score=exact_match(model(case.prompt), case.expected),
            prediction=model(case.prompt),
        )
        for case in cases
    )
    return BenchmarkReport(
        name=name,
        results=results,
        score=mean_score(result.score for result in results) if results else 0.0,
    )


SMOKE_BENCHMARK = (
    BenchmarkCase("identity", "What is the name of this model?", "Daweling"),
    BenchmarkCase("addition", "What is 2 + 2?", "4"),
)
