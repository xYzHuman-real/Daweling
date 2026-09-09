from evaluation.benchmarks import BenchmarkCase
from evaluation.experiment import ExperimentRecord, load_experiment, save_experiment
from evaluation.suites import run_suite


def test_experiment_round_trip(tmp_path):
    report = run_suite(
        "smoke",
        lambda prompt: {"2 + 2": "4"}[prompt],
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        baseline_score=0.9,
    )
    record = ExperimentRecord.from_report(report, metadata={"model": "daweling-small"})
    path = tmp_path / "experiment.json"
    save_experiment(record, path)

    loaded = load_experiment(path)
    assert loaded.name == "smoke"
    assert loaded.score == 1.0
    assert loaded.baseline_score == 0.9
    assert loaded.regression_passed is True
    assert loaded.metadata["model"] == "daweling-small"
    assert loaded.benchmarks[0]["results"][0]["prediction"] == "4"
