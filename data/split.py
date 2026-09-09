"""Deterministic train/validation splitting for Daweling JSONL datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def _bucket(example: dict, seed: int) -> int:
    key = f"{seed}\0{example.get('source','')}\0{example.get('text','')}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def split_examples(examples: Iterable[dict], validation_ratio: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Return deterministic, disjoint train/validation partitions."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    ordered = sorted(examples, key=lambda item: (_bucket(item, seed), item.get("source", ""), item.get("text", "")))
    if len(ordered) < 2:
        raise ValueError("at least two examples are required for a split")
    cutoff = max(1, min(len(ordered) - 1, round(len(ordered) * validation_ratio)))
    return ordered[cutoff:], ordered[:cutoff]


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a Daweling JSONL dataset")
    parser.add_argument("input")
    parser.add_argument("train_output")
    parser.add_argument("validation_output")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train, validation = split_examples(read_jsonl(args.input), args.validation_ratio, args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.validation_output, validation)
    print(f"train={len(train)} validation={len(validation)}")
