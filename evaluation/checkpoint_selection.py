"""Policy-driven selection of model checkpoints from evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .experiment import ExperimentRecord


@dataclass(frozen=True)
class CheckpointCandidate:
    """A checkpoint paired with its persisted evaluation record."""

    path: Path
    experiment: ExperimentRecord


@dataclass(frozen=True)
class CheckpointSelection:
    """The strongest checkpoint accepted by the selection policy."""

    selected: CheckpointCandidate
    candidates: tuple[CheckpointCandidate, ...]
    rejected: tuple[CheckpointCandidate, ...]


def select_checkpoint(
    checkpoints: Iterable[str | Path],
    evaluate: Callable[[Path], ExperimentRecord],
    *,
    require_regression_pass: bool = True,
) -> CheckpointSelection:
    """Evaluate candidate checkpoints and select the highest-scoring accepted one.

    Evaluation score is the primary ranking signal. When ``require_regression_pass``
    is enabled, candidates with an explicitly failed regression gate are rejected
    before ranking. Ties are resolved deterministically by checkpoint path.
    """
    paths = [Path(path) for path in checkpoints]
    if not paths:
        raise ValueError("at least one checkpoint is required")

    candidates = tuple(
        sorted(
            (CheckpointCandidate(path=path, experiment=evaluate(path)) for path in paths),
            key=lambda candidate: str(candidate.path),
        )
    )

    if require_regression_pass:
        accepted = tuple(
            candidate
            for candidate in candidates
            if candidate.experiment.regression_passed is not False
        )
    else:
        accepted = candidates

    if not accepted:
        raise ValueError("no checkpoint satisfies the selection policy")

    selected = max(
        accepted,
        key=lambda candidate: (candidate.experiment.score, str(candidate.path)),
    )
    rejected = tuple(candidate for candidate in candidates if candidate not in accepted)
    return CheckpointSelection(selected=selected, candidates=candidates, rejected=rejected)
