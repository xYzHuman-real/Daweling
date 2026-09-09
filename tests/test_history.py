from evaluation.experiment import ExperimentRecord
from evaluation.history import compare_experiments


def record(name: str, score: float, benchmarks=()) -> ExperimentRecord:
    return ExperimentRecord(
        name=name,
        score=score,
        baseline_score=None,
        regression_passed=None,
        benchmarks=tuple(benchmarks),
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


def test_history_reports_shared_benchmark_deltas():
    previous = record(
        "v1",
        0.60,
        [{"name": "math", "score": 0.50}, {"name": "reasoning", "score": 0.70}],
    )
    current = record(
        "v2",
        0.65,
        [{"name": "math", "score": 0.80}, {"name": "reasoning", "score": 0.60}, {"name": "new", "score": 1.0}],
    )

    comparison = compare_experiments(previous, current)
    assert [(item.name, item.delta) for item in comparison.benchmark_deltas] == [
        ("math", 0.30),
        ("reasoning", -0.10),
    ]
