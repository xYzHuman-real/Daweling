"""Utilities for comparing persisted Daweling evaluation experiments."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .experiment import ExperimentRecord, load_experiment
@dataclass(frozen=True)
class BenchmarkDelta:
    name: str
    previous_score: float
    current_score: float
    delta: float
@dataclass(frozen=True)
class ExperimentComparison:
    previous: str
    current: str
    previous_score: float
    current_score: float
    delta: float
    benchmark_deltas: tuple[BenchmarkDelta, ...]
    @property
    def improved(self) -> bool: return self.delta > 0.0
    @property
    def regressed(self) -> bool: return self.delta < 0.0
def _delta(current: float, previous: float) -> float:
    return round(current - previous, 12)
def compare_experiments(previous: ExperimentRecord, current: ExperimentRecord) -> ExperimentComparison:
    previous_benchmarks = {item["name"]: float(item["score"]) for item in previous.benchmarks}
    current_benchmarks = {item["name"]: float(item["score"]) for item in current.benchmarks}
    shared = sorted(previous_benchmarks.keys() & current_benchmarks.keys())
    deltas = tuple(BenchmarkDelta(name, previous_benchmarks[name], current_benchmarks[name], _delta(current_benchmarks[name], previous_benchmarks[name])) for name in shared)
    return ExperimentComparison(previous.name, current.name, previous.score, current.score, _delta(current.score, previous.score), deltas)
def compare_experiment_files(previous: str | Path, current: str | Path) -> ExperimentComparison:
    return compare_experiments(load_experiment(previous), load_experiment(current))
