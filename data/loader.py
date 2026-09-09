"""First-class dataset loading and deterministic partitioning for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .split import read_jsonl, split_examples


@dataclass(frozen=True)
class DatasetPartitions:
    """Text examples separated into deterministic train and validation partitions."""

    train_texts: tuple[str, ...]
    validation_texts: tuple[str, ...]
    seed: int
    validation_ratio: float

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
    )
