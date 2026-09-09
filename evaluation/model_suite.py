"""Unified model evaluation for instruction and reasoning capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .benchmarks import BenchmarkCase, BenchmarkReport, run_benchmark
from .instruction import InstructionCase, InstructionReport, run_instruction_evaluation
from .reasoning import ReasoningCase, ReasoningReport, run_reasoning_evaluation
from .regression import RegressionReport, compare_scores


@dataclass(frozen=True)
class ModelEvaluationReport:
    """Aggregate evaluation result used as the model acceptance gate."""

    name: str
    instruction: InstructionReport | None
    reasoning: ReasoningReport | None
    general: BenchmarkReport | None
    score: float
    baseline_score: float | None
    regression: RegressionReport | None

    @property
    def passed(self) -> bool:
        return self.regression is None or self.regression.passed


def evaluate_model(
    name: str,
    predict: Callable[[str], str],
    *,
    instruction_cases: Iterable[InstructionCase] = (),
    reasoning_cases: Iterable[ReasoningCase] = (),
    general_cases: Iterable[BenchmarkCase] = (),
    instruction_weight: float = 1.0,
    reasoning_weight: float = 1.0,
    general_weight: float = 1.0,
    baseline_score: float | None = None,
    regression_threshold: float = 0.0,
) -> ModelEvaluationReport:
    """Run all supplied capability evaluations and apply one aggregate regression gate."""
    instruction_list = tuple(instruction_cases)
    reasoning_list = tuple(reasoning_cases)
    general_list = tuple(general_cases)
    reports: list[tuple[float, float]] = []

    instruction = run_instruction_evaluation(name + ":instruction", instruction_list, predict) if instruction_list else None
    reasoning = run_reasoning_evaluation(name + ":reasoning", reasoning_list, predict) if reasoning_list else None
    general = run_benchmark(name + ":general", predict, general_list) if general_list else None

    for report, weight in ((instruction, instruction_weight), (reasoning, reasoning_weight), (general, general_weight)):
        if report is not None:
            if weight <= 0:
                raise ValueError("evaluation weights must be greater than zero")
            reports.append((report.score, weight))
    if not reports:
        raise ValueError("model evaluation requires at least one non-empty benchmark group")

    total_weight = sum(weight for _, weight in reports)
    score = sum(value * weight for value, weight in reports) / total_weight
    regression = compare_scores(baseline_score, score, regression_threshold) if baseline_score is not None else None
    return ModelEvaluationReport(name, instruction, reasoning, general, score, baseline_score, regression)
