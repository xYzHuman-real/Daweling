"""Versioned, reproducible manifests for Daweling datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


MANIFEST_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_example_hash(example: Mapping[str, Any]) -> str:
    """Return a stable identity for an example independent of JSON key order."""
    payload = json.dumps(dict(example), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable metadata describing one prepared dataset artifact."""

    dataset_version: str
    input_path: str
    output_path: str
    input_sha256: str
    output_sha256: str
    example_count: int
    duplicate_count: int = 0
    rejected_count: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError("unsupported dataset manifest version")
        payload["sources"] = tuple(payload.get("sources", ()))
        return cls(**payload)
