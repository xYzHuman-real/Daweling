from evaluation.benchmarks import BenchmarkCase
from evaluation.suites import BenchmarkSpec, run_suite


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


def test_weighted_suite_prioritizes_important_capabilities():
    report = run_suite(
        "weighted",
        lambda prompt: {"important": "yes", "secondary": "wrong"}[prompt],
        [
            BenchmarkSpec("important", [BenchmarkCase("a", "important", "yes")], weight=3.0),
            BenchmarkSpec("secondary", [BenchmarkCase("b", "secondary", "yes")], weight=1.0),
        ],
    )

    assert report.score == 0.75


def test_suite_report_is_json_compatible():
    report = run_suite(
        "serializable",
        lambda prompt: "4",
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        baseline_score=0.5,
    )

    snapshot = report.to_dict()
    assert snapshot["name"] == "serializable"
    assert snapshot["benchmarks"][0]["results"][0]["prediction"] == "4"
    assert snapshot["regression"]["baseline"] == 0.5
