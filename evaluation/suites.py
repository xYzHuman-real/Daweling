"""Structured evaluation suites for tracking Daweling capabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .benchmarks import BenchmarkCase, BenchmarkReport, run_benchmark
from .regression import RegressionReport, compare_scores


@dataclass(frozen=True)
class BenchmarkSpec:
    """A named benchmark group with an optional importance weight."""

    name: str
    cases: Iterable[BenchmarkCase]
    weight: float = 1.0


@dataclass(frozen=True)
class EvaluationSuiteReport:
    name: str
    benchmarks: tuple[BenchmarkReport, ...]
    score: float
    baseline_score: float | None = None
    regression: RegressionReport | None = None

    def to_dict(self) -> dict:
        """Return a JSON-compatible snapshot of the evaluation result."""
        return asdict(self)


def run_suite(
    name: str,
    model: Callable[[str], str],
    benchmarks: Iterable[tuple[str, Iterable[BenchmarkCase]] | BenchmarkSpec],
    *,
    baseline_score: float | None = None,
    regression_threshold: float = 0.0,
) -> EvaluationSuiteReport:
    """Run benchmark groups with optional weights and an aggregate regression gate.

    Legacy ``(name, cases)`` tuples remain supported. Weighted ``BenchmarkSpec``
    values make the suite score reflect the relative importance of each capability.
    """
    reports: list[BenchmarkReport] = []
    weights: list[float] = []
    for benchmark in benchmarks:
        if isinstance(benchmark, BenchmarkSpec):
            benchmark_name, cases, weight = benchmark.name, benchmark.cases, benchmark.weight
        else:
            benchmark_name, cases = benchmark
            weight = 1.0
        if weight <= 0.0:
            raise ValueError("benchmark weights must be greater than zero")
        reports.append(run_benchmark(benchmark_name, model, cases))
        weights.append(weight)

    total_weight = sum(weights)
    score = sum(report.score * weight for report, weight in zip(reports, weights)) / total_weight if reports else 0.0

    regression = None
    if baseline_score is not None:
        regression = compare_scores(baseline_score, score, regression_threshold)

    return EvaluationSuiteReport(
        name=name,
        benchmarks=tuple(reports),
        score=score,
        baseline_score=baseline_score,
        regression=regression,
    )
