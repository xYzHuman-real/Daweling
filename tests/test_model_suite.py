from evaluation.model_suite import evaluate_model
from evaluation.benchmarks import BenchmarkCase
from evaluation.instruction import InstructionCase
from evaluation.reasoning import ReasoningCase


def test_unified_model_evaluation_aggregates_capabilities():
    def predict(prompt: str) -> str:
        return {"hello": "hi", "2+2": "4", "general": "yes"}[prompt]

    report = evaluate_model(
        "smoke",
        predict,
        instruction_cases=[InstructionCase("i", "hello", "hi")],
        reasoning_cases=[ReasoningCase("r", "2+2", "4", "math")],
        general_cases=[BenchmarkCase("g", "general", "yes")],
    )
    assert report.score == 1.0
    assert report.passed
    assert report.instruction is not None
    assert report.reasoning is not None
    assert report.general is not None


def test_unified_model_evaluation_rejects_regression():
    report = evaluate_model(
        "regression",
        lambda _: "wrong",
        general_cases=[BenchmarkCase("g", "general", "yes")],
        baseline_score=1.0,
    )
    assert report.score == 0.0
    assert not report.passed
    assert report.regression is not None
