from pathlib import Path

from training.experiment import TrainingRunManifest
from training.release_pipeline import train_evaluate_release


def test_train_evaluate_release_connects_training_lineage(monkeypatch, tmp_path):
    training_manifest = TrainingRunManifest(
        run_id="run-123",
        stage="pretraining",
        dataset_manifest="dataset.json",
        dataset_sha256="dataset-sha",
        model_config={},
        training_config={},
        seed=0,
        checkpoint_path=str(tmp_path / "model.pt"),
    )

    checkpoint = tmp_path / "model.pt"
    best = tmp_path / "model.pt.best.pt"
    manifest_path = tmp_path / "model.pt.manifest.json"
    checkpoint.write_text("final", encoding="utf-8")
    best.write_text("best", encoding="utf-8")
    training_manifest.save(manifest_path)

    captured = {}

    def fake_train(*args, **kwargs):
        captured["train"] = (args, kwargs)
        return training_manifest

    def fake_release(checkpoints, benchmarks, **kwargs):
        captured["release"] = (tuple(checkpoints), tuple(benchmarks), kwargs)
        return "selection"

    monkeypatch.setattr("training.release_pipeline.train", fake_train)
    monkeypatch.setattr("training.release_pipeline.evaluate_and_select_checkpoints", fake_release)

    result = train_evaluate_release(
        tmp_path / "corpus.txt",
        checkpoint,
        10,
        3e-4,
        [("smoke", [])],
        best_output_path=best,
        experiment_dir=tmp_path / "experiments",
        release_manifest_path=tmp_path / "release.json",
    )

    assert result == (training_manifest, "selection")
    assert captured["release"][0] == (best, checkpoint)
    assert captured["release"][2]["training_manifest_path"] == manifest_path
