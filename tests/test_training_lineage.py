import json

from data.prepare import prepare_dataset


def test_prepare_module_is_importable_and_manifest_roundtrip(tmp_path):
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "clean.jsonl"
    manifest = tmp_path / "manifest.json"
    source.write_text(json.dumps({"text": " hello "}) + "\n", encoding="utf-8")
    result = prepare_dataset(source, output, manifest_path=manifest, dataset_version="v1")
    assert result.example_count == 1
    assert result.output_sha256


def test_training_run_id_is_deterministic():
    from training.experiment import make_run_id

    kwargs = {
        "stage": "pretraining",
        "dataset_sha256": "abc",
        "model_config": {"d_model": 256, "n_layers": 4},
        "training_config": {"steps": 10, "learning_rate": 0.001},
        "seed": 7,
    }
    assert make_run_id(**kwargs) == make_run_id(**kwargs)


def test_training_manifest_roundtrip(tmp_path):
    from training.experiment import TrainingRunManifest

    manifest = TrainingRunManifest(
        run_id="123",
        stage="pretraining",
        dataset_manifest="data/manifests/v1.json",
        dataset_sha256="abc",
        model_config={"d_model": 256},
        training_config={"steps": 10},
        seed=0,
        checkpoint_path="data/model.pt",
        checkpoint_sha256="def",
    )
    path = tmp_path / "run.json"
    manifest.save(path)
    assert TrainingRunManifest.load(path) == manifest
