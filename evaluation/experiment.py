"""Persistent experiment records for Daweling evaluation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .suites import EvaluationSuiteReport


@dataclass(frozen=True)
class ExperimentRecord:
    """A compact, JSON-serializable record of one evaluation experiment."""

    name: str
    score: float
    baseline_score: float | None
    regression_passed: bool | None
    benchmarks: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    @classmethod
    def from_report(
        cls,
        report: EvaluationSuiteReport,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ExperimentRecord":
        benchmarks = tuple(
            {
                "name": benchmark.name,
                "score": benchmark.score,
                "results": [
                    {
                        "case_id": result.case_id,
                        "score": result.score,
                        "prediction": result.prediction,
                    }
                    for result in benchmark.results
                ],
            }
            for benchmark in report.benchmarks
        )
        return cls(
            name=report.name,
            score=report.score,
            baseline_score=report.baseline_score,
            regression_passed=(report.regression.passed if report.regression else None),
            benchmarks=benchmarks,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_experiment(record: ExperimentRecord, path: str | Path) -> None:
    """Write an experiment record as stable, human-readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_experiment(path: str | Path) -> ExperimentRecord:
    """Load a previously saved experiment record."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment record must be a JSON object")
    benchmarks = payload.get("benchmarks", [])
    metadata = payload.get("metadata", {})
    if not isinstance(benchmarks, list) or not isinstance(metadata, dict):
        raise ValueError("Invalid experiment record structure")
    return ExperimentRecord(
        name=str(payload["name"]),
        score=float(payload["score"]),
        baseline_score=(None if payload.get("baseline_score") is None else float(payload["baseline_score"])),
        regression_passed=payload.get("regression_passed"),
        benchmarks=tuple(dict(item) for item in benchmarks),
        metadata=dict(metadata),
    )
