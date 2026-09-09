"""Persistent provenance for checkpoint release decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReleaseManifest:
    """Machine-readable link between training, evaluation, and a release candidate."""

    selected_checkpoint: str
    selected_score: float
    selected_checkpoint_sha256: str | None
    selected_experiment: str
    candidate_checkpoints: tuple[str, ...]
    rejected_checkpoints: tuple[str, ...]
    training_run_id: str | None = None
    dataset_sha256: str | None = None
    evaluation_experiments: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReleaseManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Release manifest must be a JSON object")
        return cls(
            selected_checkpoint=str(payload["selected_checkpoint"]),
            selected_score=float(payload["selected_score"]),
            selected_checkpoint_sha256=payload.get("selected_checkpoint_sha256"),
            selected_experiment=str(payload["selected_experiment"]),
            candidate_checkpoints=tuple(payload.get("candidate_checkpoints", [])),
            rejected_checkpoints=tuple(payload.get("rejected_checkpoints", [])),
            training_run_id=payload.get("training_run_id"),
            dataset_sha256=payload.get("dataset_sha256"),
            evaluation_experiments=dict(payload.get("evaluation_experiments", {})),
            metadata=dict(payload.get("metadata", {})),
        )
