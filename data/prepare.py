"""Data preparation with validation, normalization, deterministic deduplication, and manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .manifest import DatasetManifest, canonical_example_hash, sha256_file
except ImportError:  # Support ``python data/prepare.py ...`` from the repository root.
    from data.manifest import DatasetManifest, canonical_example_hash, sha256_file


def validate_example(example: Any) -> tuple[bool, str]:
    if not isinstance(example, dict):
        return False, "example must be an object"
    text = example.get("text")
    if not isinstance(text, str):
        return False, "text must be a string"
    if not text.strip():
        return False, "text must not be empty"
    source = example.get("source")
    if source is not None and not isinstance(source, str):
        return False, "source must be a string when provided"
    return True, "ok"


def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(example)
    normalized["text"] = normalized["text"].strip()
    if isinstance(normalized.get("source"), str):
        normalized["source"] = normalized["source"].strip()
    return normalized


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    dataset_version: str = "v1",
    sources: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DatasetManifest:
    """Validate, normalize, and deduplicate a JSONL dataset."""
    source_path, destination = Path(input_path), Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    valid = duplicates = invalid = 0

    with source_path.open("r", encoding="utf-8") as source, destination.open("w", encoding="utf-8") as target:
        for line in source:
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            ok, _ = validate_example(example)
            if not ok:
                invalid += 1
                continue
            normalized = normalize_example(example)
            fingerprint = canonical_example_hash(normalized)
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            target.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
            valid += 1

    manifest = DatasetManifest(
        dataset_version=dataset_version,
        input_path=str(source_path),
        output_path=str(destination),
        input_sha256=sha256_file(source_path),
        output_sha256=sha256_file(destination),
        example_count=valid,
        duplicate_count=duplicates,
        invalid_count=invalid,
        preprocessing={"normalize": True, "deduplicate": True, "validation": True},
        sources=sources or [],
        metadata=metadata or {},
    )
    if manifest_path is not None:
        manifest.save(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Daweling JSONL dataset")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()
    manifest = prepare_dataset(args.input, args.output, manifest_path=args.manifest, dataset_version=args.version, sources=args.source)
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
