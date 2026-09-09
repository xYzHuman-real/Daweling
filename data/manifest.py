"""Dataset provenance, fingerprints, and reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

MANIFEST_VERSION = "1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_example_hash(example: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(example), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    """Machine-readable record of a prepared dataset and its lineage."""

    dataset_version: str
    input_path: str
    output_path: str
    input_sha256: str
    output_sha256: str
    example_count: int
    duplicate_count: int
    invalid_count: int
    split_counts: dict[str, int] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_rejected_count: int = 0
    contamination_rejected_count: int = 0
    quality_issues: dict[str, int] = field(default_factory=dict)
    contamination_benchmark_ids: dict[str, int] = field(default_factory=dict)
    manifest_version: str = MANIFEST_VERSION

    def __post_init__(self) -> None:
        for name in (
            "example_count",
            "duplicate_count",
            "invalid_count",
            "quality_rejected_count",
            "contamination_rejected_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
