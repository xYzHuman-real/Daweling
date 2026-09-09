import json
from pathlib import Path

from data.manifest import DatasetManifest, canonical_example_hash, sha256_file
from data.prepare import prepare_dataset, validate_example


def test_validation_rejects_bad_examples():
    assert validate_example({"text": "hello", "source": "test"}) == (True, "ok")
    assert validate_example({"text": ""})[0] is False
    assert validate_example({"text": 123})[0] is False
    assert validate_example("hello")[0] is False


def test_canonical_hash_is_order_independent():
    assert canonical_example_hash({"text": "hello", "source": "x"}) == canonical_example_hash({"source": "x", "text": "hello"})


def test_prepare_deduplicates_invalid_and_normalizes(tmp_path):
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "clean.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rows = [
        {"text": " hello ", "source": " test "},
        {"source": "test", "text": "hello"},
        {"text": ""},
        {"text": "world"},
        "not an object",
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = prepare_dataset(source, output, manifest_path=manifest_path, dataset_version="v2", sources=["test-source"])

    assert manifest.example_count == 2
    assert manifest.duplicate_count == 1
    assert manifest.invalid_count == 2
    assert manifest.input_sha256 == sha256_file(source)
    assert manifest.output_sha256 == sha256_file(output)
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0]) == {"source": "test", "text": "hello"}


def test_manifest_round_trip(tmp_path):
    manifest = DatasetManifest("v1", "raw", "clean", "a", "b", 2, 1, 3, {"train": 2}, {"deduplicate": True}, ["source"], {"owner": "daweling"})
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert DatasetManifest.load(path) == manifest
