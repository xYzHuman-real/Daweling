"""Policy-driven selection of model checkpoints from evaluation results."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from .experiment import ExperimentRecord
@dataclass(frozen=True)
class CheckpointCandidate:
    path: Path
    experiment: ExperimentRecord
@dataclass(frozen=True)
class CheckpointSelection:
    selected: CheckpointCandidate
    candidates: tuple[CheckpointCandidate, ...]
    rejected: tuple[CheckpointCandidate, ...]
def select_checkpoint(checkpoints: Iterable[str | Path], evaluate: Callable[[Path], ExperimentRecord], *, require_regression_pass: bool = True) -> CheckpointSelection:
    paths = [Path(p) for p in checkpoints]
    if not paths: raise ValueError("at least one checkpoint is required")
    candidates = tuple(sorted((CheckpointCandidate(p, evaluate(p)) for p in paths), key=lambda c: str(c.path)))
    if require_regression_pass:
        accepted = tuple(c for c in candidates if c.experiment.regression_passed is not False and c.experiment.score > 0.0)
    else:
        accepted = tuple(c for c in candidates if c.experiment.score > 0.0)
    if not accepted: raise ValueError("no checkpoint satisfies the selection policy")
    selected = max(accepted, key=lambda c: (c.experiment.score, str(c.path)))
    rejected = tuple(c for c in candidates if c not in accepted)
    return CheckpointSelection(selected, candidates, rejected)
