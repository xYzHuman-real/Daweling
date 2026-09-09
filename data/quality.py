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
    if not isinstance(text, str): return QualityResult(False, "", (QualityIssue("schema", "text must be a string"),))
    normalized = text.strip()
    if len(normalized) < min_chars: issues.append(QualityIssue("too_short", f"text is shorter than {min_chars} characters"))
    if len(normalized) > max_chars: issues.append(QualityIssue("too_long", f"text exceeds {max_chars} characters"))
    if normalized and normalized.count("\ufffd") / len(normalized) > 0.01: issues.append(QualityIssue("encoding_noise", "text contains excessive replacement characters"))
    return QualityResult(not issues, normalized, tuple(issues))
def find_duplicate_texts(texts: Iterable[str]) -> dict[str, tuple[int, ...]]:
    positions: dict[str, list[int]] = {}
    for index, text in enumerate(texts):
        fingerprint = hashlib.sha256(normalize_for_quality(text).encode("utf-8")).hexdigest()
        positions.setdefault(fingerprint, []).append(index)
    return {f: tuple(i) for f, i in positions.items() if len(i) > 1}
def find_benchmark_contamination(examples: Iterable[Mapping[str, object]], benchmarks: Iterable[Mapping[str, object]]) -> tuple[ContaminationMatch, ...]:
    """Find benchmark prompt or expected-answer text embedded in training examples."""
    benchmark_parts: dict[str, tuple[str, str]] = {}
    for benchmark in benchmarks:
        bid = str(benchmark.get("id", ""))
        if bid:
            benchmark_parts[bid] = (normalize_for_quality(str(benchmark.get("prompt", ""))), normalize_for_quality(str(benchmark.get("expected", ""))))
    matches: list[ContaminationMatch] = []
    for index, example in enumerate(examples):
        text = normalize_for_quality(str(example.get("text", "")))
        if not text: continue
        for bid, (prompt, expected) in benchmark_parts.items():
            matched = prompt if prompt and prompt in text else expected if expected and expected in text else ""
            if matched: matches.append(ContaminationMatch(index, bid, matched))
    return tuple(matches)
