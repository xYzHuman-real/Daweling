"""Reproducible training-run lineage and checkpoint metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingRunManifest:
    """Machine-readable link between data, configuration, and a checkpoint."""

    run_id: str
    stage: str
    dataset_manifest: str | None
    dataset_sha256: str | None
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    seed: int
    checkpoint_path: str
    checkpoint_sha256: str | None = None
    parent_checkpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    last_step: int = 0
    best_validation_loss: float | None = None
    best_step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingRunManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run_id(*, stage: str, dataset_sha256: str | None, model_config: dict[str, Any], training_config: dict[str, Any], seed: int) -> str:
    payload = json.dumps(
        {"stage": stage, "dataset_sha256": dataset_sha256, "model_config": model_config, "training_config": training_config, "seed": seed},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
