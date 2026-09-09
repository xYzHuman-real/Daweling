"""Prepare newline-delimited JSON examples for Daweling model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .manifest import DatasetManifest, canonical_example_hash, sha256_file

REQUIRED_FIELDS = {"text", "source"}


def validate_example(example: object) -> tuple[bool, str]:
    if not isinstance(example, dict):
        return False, "example must be an object"
    missing = REQUIRED_FIELDS - set(example)
    if missing:
        return False, f"missing fields: {', '.join(sorted(missing))}"
    if not isinstance(example["text"], str) or not example["text"].strip():
        return False, "text must be a non-empty string"
    if not isinstance(example["source"], str) or not example["source"].strip():
        return False, "source must be a non-empty string"
    if "quality" in example and not isinstance(example["quality"], (int, float)):
        return False, "quality must be numeric"
    if "quality" in example and not 0 <= float(example["quality"]) <= 1:
        return False, "quality must be between 0 and 1"
    return True, "ok"


def normalize_example(example: dict) -> dict:
    normalized = dict(example)
    normalized["text"] = normalized["text"].strip()
    normalized["source"] = normalized["source"].strip()
    return normalized


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            valid, reason = validate_example(example)
            if not valid:
                raise ValueError(f"Invalid example on line {line_number}: {reason}")
            yield normalize_example(example)


def prepare_dataset(input_path: str | Path, output_path: str | Path, *, dataset_version: str = "v1", manifest_path: str | Path | None = None, metadata: dict | None = None) -> DatasetManifest:
    input_path = Path(input_path)
    output_path = Path(output_path)
    seen: set[str] = set()
    examples: list[dict] = []
    duplicate_count = 0

    for example in read_jsonl(input_path):
        identity = canonical_example_hash(example)
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        examples.append(example)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")

    sources = tuple(sorted({example["source"] for example in examples}))
    manifest = DatasetManifest(
        dataset_version=dataset_version,
        input_path=str(input_path),
        output_path=str(output_path),
        input_sha256=sha256_file(input_path),
        output_sha256=sha256_file(output_path),
        example_count=len(examples),
        duplicate_count=duplicate_count,
        preprocessing={"normalize_whitespace": True, "deduplicate": True, "identity": "sha256(canonical_json)"},
        sources=sources,
        metadata=metadata or {},
    )
    if manifest_path is not None:
        manifest.save(manifest_path)
    return manifest


def write_clean_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    """Backward-compatible validation/cleaning entry point."""
    return prepare_dataset(input_path, output_path).example_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate, normalize and deduplicate a Daweling JSONL dataset")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    manifest = prepare_dataset(args.input, args.output, dataset_version=args.version, manifest_path=args.manifest)
    print(f"Prepared {manifest.example_count} examples; removed {manifest.duplicate_count} duplicates")
