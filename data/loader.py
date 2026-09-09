"""First-class dataset loading and deterministic partitioning for training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .split import read_jsonl, split_examples, write_jsonl


def _rows_sha256(rows: list[dict]) -> str:
    payload = "".join(f"{row!r}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DatasetPartitions:
    """Text examples plus preserved metadata for deterministic training policies."""

    train_texts: tuple[str, ...]
    validation_texts: tuple[str, ...]
    seed: int
    validation_ratio: float
    train_sha256: str
    validation_sha256: str
    train_rows: tuple[dict, ...] = ()
    validation_rows: tuple[dict, ...] = ()

    @property
    def train_count(self) -> int:
        return len(self.train_texts)

    @property
    def validation_count(self) -> int:
        return len(self.validation_texts)


def load_partitions(path: str | Path, *, validation_ratio: float = 0.1, seed: int = 0) -> DatasetPartitions:
    """Load a prepared JSONL dataset and create deterministic train/validation partitions."""
    rows = read_jsonl(path)
    train, validation = split_examples(rows, validation_ratio=validation_ratio, seed=seed)
    return DatasetPartitions(
        train_texts=tuple(row["text"] for row in train),
        validation_texts=tuple(row["text"] for row in validation),
        seed=seed,
        validation_ratio=validation_ratio,
        train_sha256=_rows_sha256(train),
        validation_sha256=_rows_sha256(validation),
        train_rows=tuple(train),
        validation_rows=tuple(validation),
    )
