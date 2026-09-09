"""Data preparation with validation, quality filtering, deduplication, and manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .manifest import DatasetManifest, canonical_example_hash, sha256_file
    from .quality import find_benchmark_contamination, quality_check
except ImportError:  # Support ``python data/prepare.py ...`` from the repository root.
    from data.manifest import DatasetManifest, canonical_example_hash, sha256_file
    from data.quality import find_benchmark_contamination, quality_check


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


def _load_benchmarks(path: str | Path | None) -> list[Mapping[str, object]]:
    if path is None:
        return []
    rows: list[Mapping[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    dataset_version: str = "v1",
    sources: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    min_chars: int = 1,
    max_chars: int = 200_000,
    contamination_benchmarks: Iterable[Mapping[str, object]] | None = None,
) -> DatasetManifest:
    """Validate, quality-filter, normalize, optionally contamination-filter, and deduplicate JSONL."""
    source_path, destination = Path(input_path), Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("quality limits must satisfy 0 <= min_chars <= max_chars")

    raw_examples: list[dict[str, Any]] = []
    invalid = 0
    with source_path.open("r", encoding="utf-8") as source:
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
            raw_examples.append(normalize_example(example))

    quality_rejected = 0
    quality_issues: Counter[str] = Counter()
    quality_examples: list[dict[str, Any]] = []
    for example in raw_examples:
        result = quality_check(example["text"], min_chars=min_chars, max_chars=max_chars)
        if not result.accepted:
            quality_rejected += 1
            quality_issues.update(issue.kind for issue in result.issues)
            continue
        example["text"] = result.normalized_text
        quality_examples.append(example)

    benchmarks = list(contamination_benchmarks or [])
    contamination_matches = find_benchmark_contamination(quality_examples, benchmarks)
    contaminated_indices = {match.example_index for match in contamination_matches}
    contamination_ids: Counter[str] = Counter(match.benchmark_id for match in contamination_matches)

    seen: set[str] = set()
    valid = duplicates = 0
    with destination.open("w", encoding="utf-8") as target:
        for index, example in enumerate(quality_examples):
            if index in contaminated_indices:
                continue
            fingerprint = canonical_example_hash(example)
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            target.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
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
        preprocessing={
            "normalize": True,
            "deduplicate": True,
            "validation": True,
            "quality_filter": True,
            "contamination_filter": bool(benchmarks),
            "min_chars": min_chars,
            "max_chars": max_chars,
        },
        sources=sources or [],
        metadata=metadata or {},
        quality_rejected_count=quality_rejected,
        contamination_rejected_count=len(contaminated_indices),
        quality_issues=dict(sorted(quality_issues.items())),
        contamination_benchmark_ids=dict(sorted(contamination_ids.items())),
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
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--benchmark-jsonl", type=Path, help="Optional JSONL benchmark cases for exact contamination filtering")
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.input,
        args.output,
        manifest_path=args.manifest,
        dataset_version=args.version,
        sources=args.source,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        contamination_benchmarks=_load_benchmarks(args.benchmark_jsonl),
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
