"""Data-quality and evaluation-contamination checks for Daweling datasets."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class QualityIssue:
    kind: str
    message: str


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    normalized_text: str
    issues: tuple[QualityIssue, ...] = ()


@dataclass(frozen=True)
class ContaminationMatch:
    example_index: int
    benchmark_id: str
    matched_text: str


def normalize_for_quality(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def quality_check(text: str, *, min_chars: int = 1, max_chars: int = 200_000) -> QualityResult:
    issues: list[QualityIssue] = []
    if not isinstance(text, str):
        return QualityResult(False, "", (QualityIssue("schema", "text must be a string"),))
    normalized = text.strip()
    if len(normalized) < min_chars:
        issues.append(QualityIssue("too_short", f"text is shorter than {min_chars} characters"))
    if len(normalized) > max_chars:
        issues.append(QualityIssue("too_long", f"text exceeds {max_chars} characters"))
    if normalized and normalized.count("\ufffd") / len(normalized) > 0.01:
        issues.append(QualityIssue("encoding_noise", "text contains excessive replacement characters"))
    return QualityResult(not issues, normalized, tuple(issues))


def find_duplicate_texts(texts: Iterable[str]) -> dict[str, tuple[int, ...]]:
    positions: dict[str, list[int]] = {}
    for index, text in enumerate(texts):
        fingerprint = hashlib.sha256(normalize_for_quality(text).encode("utf-8")).hexdigest()
        positions.setdefault(fingerprint, []).append(index)
    return {fingerprint: tuple(indices) for fingerprint, indices in positions.items() if len(indices) > 1}


def find_benchmark_contamination(examples: Iterable[Mapping[str, object]], benchmarks: Iterable[Mapping[str, object]]) -> tuple[ContaminationMatch, ...]:
    """Find exact normalized benchmark prompt/expected-text matches in training examples."""
    benchmark_texts: dict[str, str] = {}
    for benchmark in benchmarks:
        benchmark_id = str(benchmark.get("id", ""))
        if not benchmark_id:
            continue
        parts = [str(benchmark.get("prompt", "")), str(benchmark.get("expected", ""))]
        benchmark_texts[benchmark_id] = normalize_for_quality(" ".join(part for part in parts if part))
    matches: list[ContaminationMatch] = []
    for index, example in enumerate(examples):
        text = normalize_for_quality(str(example.get("text", "")))
        if not text:
            continue
        for benchmark_id, benchmark_text in benchmark_texts.items():
            if benchmark_text and benchmark_text in text:
                matches.append(ContaminationMatch(index, benchmark_id, benchmark_text))
    return tuple(matches)
