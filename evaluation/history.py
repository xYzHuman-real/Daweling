"""Utilities for comparing persisted Daweling evaluation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .experiment import ExperimentRecord, load_experiment


@dataclass(frozen=True)
class ExperimentComparison:
    """Score delta between two persisted experiments."""

    previous: str
    current: str
    previous_score: float
    current_score: float
    delta: float

    @property
    def improved(self) -> bool:
        return self.delta > 0.0

    @property
    def regressed(self) -> bool:
        return self.delta < 0.0


def compare_experiments(previous: ExperimentRecord, current: ExperimentRecord) -> ExperimentComparison:
    """Compare two experiment records without requiring the same benchmark set."""
    return ExperimentComparison(
        previous=previous.name,
        current=current.name,
        previous_score=previous.score,
        current_score=current.score,
        delta=current.score - previous.score,
    )


def compare_experiment_files(previous: str | Path, current: str | Path) -> ExperimentComparison:
    """Load and compare two JSON experiment records."""
    return compare_experiments(load_experiment(previous), load_experiment(current))
