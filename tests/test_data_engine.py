import json

from data.manifest import DatasetManifest, canonical_example_hash, sha256_file
from data.prepare import prepare_dataset


def test_canonical_hash_is_independent_of_key_order():
    assert canonical_example_hash({"text": "hello", "source": "x"}) == canonical_example_hash({"source": "x", "text": "hello"})


def test_prepare_normalizes_and_deduplicates(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "clean.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rows = [
        {"source": " test ", "text": " hello "},
        {"text": "hello", "source": "test"},
        {"text": "world", "source": "test"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = prepare_dataset(source, output, dataset_version="train-v1", manifest_path=manifest_path)

    assert manifest.example_count == 2
    assert manifest.duplicate_count == 1
    assert manifest.sources == ("test",)
    assert output.read_text(encoding="utf-8").count("\n") == 2
    assert manifest.input_sha256 == sha256_file(source)
    assert DatasetManifest.load(manifest_path) == manifest


def test_prepare_rejects_invalid_examples(tmp_path):
    source = tmp_path / "bad.jsonl"
    output = tmp_path / "clean.jsonl"
    source.write_text(json.dumps({"source": "test", "text": ""}) + "\n", encoding="utf-8")

    try:
        prepare_dataset(source, output)
    except ValueError as exc:
        assert "Invalid example" in str(exc)
    else:
        raise AssertionError("invalid dataset should be rejected")
