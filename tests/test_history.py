from evaluation.experiment import ExperimentRecord
from evaluation.history import compare_experiments


def record(name: str, score: float) -> ExperimentRecord:
    return ExperimentRecord(
        name=name,
        score=score,
        baseline_score=None,
        regression_passed=None,
        benchmarks=(),
        metadata={},
    )


def test_history_detects_improvement():
    comparison = compare_experiments(record("v1", 0.60), record("v2", 0.75))
    assert comparison.delta == 0.15
    assert comparison.improved is True
    assert comparison.regressed is False


def test_history_detects_regression():
    comparison = compare_experiments(record("v1", 0.75), record("v2", 0.70))
    assert comparison.delta == -0.05
    assert comparison.improved is False
    assert comparison.regressed is True
