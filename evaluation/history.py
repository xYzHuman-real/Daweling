"""Utilities for comparing persisted Daweling evaluation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .experiment import ExperimentRecord, load_experiment


@dataclass(frozen=True)
class BenchmarkDelta:
    """Score change for one benchmark shared by two experiments."""

    name: str
    previous_score: float
    current_score: float
    delta: float


@dataclass(frozen=True)
class ExperimentComparison:
    """Score and benchmark-level deltas between two experiments."""

    previous: str
    current: str
    previous_score: float
    current_score: float
    delta: float
    benchmark_deltas: tuple[BenchmarkDelta, ...]

    @property
    def improved(self) -> bool:
        return self.delta > 0.0

    @property
    def regressed(self) -> bool:
        return self.delta < 0.0


def compare_experiments(previous: ExperimentRecord, current: ExperimentRecord) -> ExperimentComparison:
    """Compare overall scores and shared benchmarks between two experiments."""
    previous_benchmarks = {item["name"]: float(item["score"]) for item in previous.benchmarks}
    current_benchmarks = {item["name"]: float(item["score"]) for item in current.benchmarks}
    shared = sorted(previous_benchmarks.keys() & current_benchmarks.keys())
    deltas = tuple(
        BenchmarkDelta(
            name=name,
            previous_score=previous_benchmarks[name],
            current_score=current_benchmarks[name],
            delta=current_benchmarks[name] - previous_benchmarks[name],
        )
        for name in shared
    )
    return ExperimentComparison(
        previous=previous.name,
        current=current.name,
        previous_score=previous.score,
        current_score=current.score,
        delta=current.score - previous.score,
        benchmark_deltas=deltas,
    )


def compare_experiment_files(previous: str | Path, current: str | Path) -> ExperimentComparison:
    """Load and compare two JSON experiment records."""
    return compare_experiments(load_experiment(previous), load_experiment(current))
