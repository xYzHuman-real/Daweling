"""Structured evaluation suites for tracking Daweling capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .benchmarks import BenchmarkCase, BenchmarkReport, run_benchmark
from .regression import RegressionReport, compare_scores


@dataclass(frozen=True)
class EvaluationSuiteReport:
    name: str
    benchmarks: tuple[BenchmarkReport, ...]
    score: float
    baseline_score: float | None = None
    regression: RegressionReport | None = None


def run_suite(
    name: str,
    model: Callable[[str], str],
    benchmarks: Iterable[tuple[str, Iterable[BenchmarkCase]]],
    *,
    baseline_score: float | None = None,
    regression_threshold: float = 0.0,
) -> EvaluationSuiteReport:
    """Run multiple benchmark groups and optionally enforce a regression gate."""
    reports = tuple(run_benchmark(benchmark_name, model, cases) for benchmark_name, cases in benchmarks)
    score = sum(report.score for report in reports) / len(reports) if reports else 0.0

    regression = None
    if baseline_score is not None:
        regression = compare_scores(baseline_score, score, regression_threshold)

    return EvaluationSuiteReport(
        name=name,
        benchmarks=reports,
        score=score,
        baseline_score=baseline_score,
        regression=regression,
    )
