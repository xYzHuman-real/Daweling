from evaluation.benchmarks import BenchmarkCase
from evaluation.suites import run_suite


def test_suite_aggregates_benchmark_scores():
    benchmarks = [
        ("math", [BenchmarkCase("addition", "2 + 2", "4")]),
        ("identity", [BenchmarkCase("name", "model", "Daweling")]),
    ]

    report = run_suite(
        "smoke",
        lambda prompt: {"2 + 2": "4", "model": "Daweling"}[prompt],
        benchmarks,
    )

    assert report.score == 1.0
    assert len(report.benchmarks) == 2
    assert report.regression is None


def test_suite_exposes_regression_gate():
    report = run_suite(
        "regression",
        lambda prompt: "wrong",
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        baseline_score=1.0,
        regression_threshold=0.1,
    )

    assert report.regression is not None
    assert report.regression.passed is False
