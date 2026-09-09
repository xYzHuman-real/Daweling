"""Persistent registry for evaluated Daweling model checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .experiment import ExperimentRecord, load_experiment


@dataclass(frozen=True)
class ModelRegistryEntry:
    """A checkpoint promoted into the model-development registry."""

    checkpoint: str
    experiment: str
    score: float
    regression_passed: bool | None
    metadata: dict[str, Any]

    @classmethod
    def from_record(cls, checkpoint: str | Path, record: ExperimentRecord) -> "ModelRegistryEntry":
        return cls(
            checkpoint=str(checkpoint),
            experiment=record.name,
            score=record.score,
            regression_passed=record.regression_passed,
            metadata=dict(record.metadata),
        )


@dataclass(frozen=True)
class RegistrySnapshot:
    """Complete persisted registry state."""

    best: ModelRegistryEntry | None
    history: tuple[ModelRegistryEntry, ...]


class ModelRegistry:
    """Track evaluated checkpoints and promote the best passing candidate."""

    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RegistrySnapshot:
        if not self.path.exists():
            return RegistrySnapshot(best=None, history=())
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != self.VERSION:
            raise ValueError("Unsupported model registry format")
        raw_history = payload.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError("Invalid model registry history")
        history = tuple(self._entry(item) for item in raw_history)
        raw_best = payload.get("best")
        best = self._entry(raw_best) if raw_best is not None else None
        return RegistrySnapshot(best=best, history=history)

    def register(
        self,
        checkpoint: str | Path,
        record: ExperimentRecord | str | Path,
        *,
        require_regression_pass: bool = True,
    ) -> ModelRegistryEntry:
        """Record an experiment and promote it when it is a valid improvement."""
        experiment = load_experiment(record) if isinstance(record, (str, Path)) else record
        entry = ModelRegistryEntry.from_record(checkpoint, experiment)
        if entry.score < 0.0 or entry.score > 1.0:
            raise ValueError("model score must be between 0 and 1")
        if require_regression_pass and entry.regression_passed is False:
            raise ValueError("cannot register a regression-failing model")

        snapshot = self.load()
        history = snapshot.history + (entry,)
        best = snapshot.best
        if best is None or entry.score > best.score:
            best = entry
        self._save(RegistrySnapshot(best=best, history=history))
        return entry

    def best(self) -> ModelRegistryEntry | None:
        """Return the currently promoted best checkpoint."""
        return self.load().best

    def history(self) -> tuple[ModelRegistryEntry, ...]:
        """Return registry entries in registration order."""
        return self.load().history

    def _save(self, snapshot: RegistrySnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "best": asdict(snapshot.best) if snapshot.best is not None else None,
            "history": [asdict(entry) for entry in snapshot.history],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _entry(payload: Any) -> ModelRegistryEntry:
        if not isinstance(payload, dict):
            raise ValueError("Invalid model registry entry")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Invalid model registry metadata")
        return ModelRegistryEntry(
            checkpoint=str(payload["checkpoint"]),
            experiment=str(payload["experiment"]),
            score=float(payload["score"]),
            regression_passed=payload.get("regression_passed"),
            metadata=dict(metadata),
        )
