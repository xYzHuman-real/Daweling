from pathlib import Path

from evaluation.benchmarks import BenchmarkCase
from evaluation.experiment import load_experiment
from evaluation.model_registry import ModelRegistry
from evaluation.release import evaluate_and_select_checkpoints
from evaluation.release_manifest import ReleaseManifest


def test_release_pipeline_evaluates_all_candidates_and_selects_best(monkeypatch, tmp_path):
    scores = {"data/a.pt": "wrong", "data/b.pt": "4"}
    for path in (Path("data/a.pt"), Path("data/b.pt")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    def fake_generate(path, prompt, *, config, device):
        return scores[str(path)]

    monkeypatch.setattr("evaluation.checkpoint_evaluator.generate_from_checkpoint", fake_generate)

    selection = evaluate_and_select_checkpoints(
        [Path("data/a.pt"), Path("data/b.pt")],
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        experiment_dir=tmp_path,
    )

    assert selection.selected.path == Path("data/b.pt")
    assert len(selection.candidates) == 2
    assert load_experiment(tmp_path / "a.json").score == 0.0
    assert load_experiment(tmp_path / "b.json").score == 1.0


def test_release_pipeline_rejects_regression_failures(monkeypatch, tmp_path):
    scores = {"data/a.pt": "4", "data/b.pt": "wrong"}
    for path in (Path("data/a.pt"), Path("data/b.pt")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    def fake_generate(path, prompt, *, config, device):
        return scores[str(path)]

    monkeypatch.setattr("evaluation.checkpoint_evaluator.generate_from_checkpoint", fake_generate)

    selection = evaluate_and_select_checkpoints(
        [Path("data/a.pt"), Path("data/b.pt")],
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        baseline_score=1.0,
        regression_threshold=0.0,
    )

    assert selection.selected.path == Path("data/a.pt")
    assert [candidate.path for candidate in selection.rejected] == [Path("data/b.pt")]


def test_release_manifest_links_evaluation_to_training_lineage(monkeypatch, tmp_path):
    scores = {"data/a.pt": "4", "data/b.pt": "wrong"}

    def fake_generate(path, prompt, *, config, device):
        return scores[str(path)]

    monkeypatch.setattr("evaluation.checkpoint_evaluator.generate_from_checkpoint", fake_generate)
    for path in (Path("data/a.pt"), Path("data/b.pt")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    training_manifest = tmp_path / "training.json"
    training_manifest.write_text(
        '{"run_id":"run-123","stage":"pretraining","dataset_manifest":"data/dataset.json",'
        '"dataset_sha256":"dataset-hash","model_config":{},"training_config":{},"seed":0,'
        '"checkpoint_path":"data/a.pt"}',
        encoding="utf-8",
    )
    release_path = tmp_path / "release.json"

    evaluate_and_select_checkpoints(
        [Path("data/a.pt"), Path("data/b.pt")],
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        training_manifest_path=training_manifest,
        release_manifest_path=release_path,
    )

    manifest = ReleaseManifest.load(release_path)
    assert manifest.selected_checkpoint == "data/a.pt"
    assert manifest.training_run_id == "run-123"
    assert manifest.dataset_sha256 == "dataset-hash"
    assert manifest.rejected_checkpoints == ("data/b.pt",)
    assert manifest.evaluation_experiments["data/a.pt"] == "a"


def test_release_pipeline_promotes_selected_model_to_registry(monkeypatch, tmp_path):
    scores = {"data/a.pt": "wrong", "data/b.pt": "4"}

    def fake_generate(path, prompt, *, config, device):
        return scores[str(path)]

    monkeypatch.setattr("evaluation.checkpoint_evaluator.generate_from_checkpoint", fake_generate)
    for path in (Path("data/a.pt"), Path("data/b.pt")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    registry_path = tmp_path / "models.json"
    selection = evaluate_and_select_checkpoints(
        [Path("data/a.pt"), Path("data/b.pt")],
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        experiment_dir=tmp_path / "experiments",
        model_registry_path=registry_path,
    )

    best = ModelRegistry(registry_path).best()
    assert best is not None
    assert best.checkpoint == str(selection.selected.path)
    assert best.score == 1.0
