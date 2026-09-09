"""Prepare newline-delimited JSON examples for Daweling model training.

This utility deliberately performs lightweight, dependency-free validation. It
is not a substitute for dataset-specific quality, licensing, or safety review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = {"text", "source"}


def validate_example(example: object) -> tuple[bool, str]:
    if not isinstance(example, dict):
        return False, "example must be an object"
    missing = REQUIRED_FIELDS - set(example)
    if missing:
        return False, f"missing fields: {', '.join(sorted(missing))}"
    if not isinstance(example["text"], str) or not example["text"].strip():
        return False, "text must be a non-empty string"
    if not isinstance(example["source"], str) or not example["source"].strip():
        return False, "source must be a non-empty string"
    if "quality" in example and not isinstance(example["quality"], (int, float)):
        return False, "quality must be numeric"
    if "quality" in example and not 0 <= float(example["quality"]) <= 1:
        return False, "quality must be between 0 and 1"
    return True, "ok"


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            valid, reason = validate_example(example)
            if not valid:
                raise ValueError(f"Invalid example on line {line_number}: {reason}")
            yield example


def write_clean_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    count = 0
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for example in read_jsonl(input_path):
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate and copy a Daweling JSONL dataset")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    print(f"Validated {write_clean_jsonl(args.input, args.output)} examples")
