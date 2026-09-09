from evaluation.experiment import ExperimentRecord
from evaluation.model_registry import ModelRegistry


def record(name: str, score: float, regression_passed: bool | None = None) -> ExperimentRecord:
    return ExperimentRecord(
        name=name,
        score=score,
        baseline_score=None,
        regression_passed=regression_passed,
        benchmarks=(),
        metadata={"model": "daweling-small"},
    )


def test_registry_promotes_best_checkpoint(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")

    registry.register("model-v1.pt", record("v1", 0.60))
    registry.register("model-v2.pt", record("v2", 0.75))
    registry.register("model-v3.pt", record("v3", 0.70))

    assert registry.best() is not None
    assert registry.best().checkpoint == "model-v2.pt"
    assert registry.best().score == 0.75
    assert [entry.experiment for entry in registry.history()] == ["v1", "v2", "v3"]


def test_registry_persists_and_reloads(tmp_path):
    path = tmp_path / "registry.json"
    ModelRegistry(path).register("model.pt", record("v1", 0.8))

    reloaded = ModelRegistry(path)
    assert reloaded.best() is not None
    assert reloaded.best().checkpoint == "model.pt"
    assert reloaded.best().metadata["model"] == "daweling-small"


def test_registry_rejects_failed_regression(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")

    try:
        registry.register("bad.pt", record("bad", 0.9, regression_passed=False))
    except ValueError as exc:
        assert "regression-failing" in str(exc)
    else:
        raise AssertionError("expected regression-failing checkpoint to be rejected")


def test_registry_can_load_experiment_file(tmp_path):
    from evaluation.experiment import save_experiment

    experiment_path = tmp_path / "experiment.json"
    save_experiment(record("file-run", 0.65), experiment_path)

    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("file.pt", experiment_path)
    assert registry.best().experiment == "file-run"
