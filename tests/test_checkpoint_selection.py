from pathlib import Path

import pytest

from evaluation.checkpoint_selection import select_checkpoint
from evaluation.experiment import ExperimentRecord


def record(name: str, score: float, regression_passed: bool | None = None) -> ExperimentRecord:
    return ExperimentRecord(
        name=name,
        score=score,
        baseline_score=None,
        regression_passed=regression_passed,
        benchmarks=(),
        metadata={},
    )


def test_selects_highest_scoring_accepted_checkpoint():
    scores = {
        Path("data/v1.pt"): record("v1", 0.70, True),
        Path("data/v2.pt"): record("v2", 0.85, True),
        Path("data/v3.pt"): record("v3", 0.80, None),
    }

    selection = select_checkpoint(scores, lambda path: scores[path])

    assert selection.selected.path == Path("data/v2.pt")
    assert selection.selected.experiment.score == 0.85
    assert selection.rejected == ()


def test_rejects_failed_regression_gate_by_default():
    scores = {
        Path("data/v1.pt"): record("v1", 0.70, True),
        Path("data/v2.pt"): record("v2", 0.95, False),
    }

    selection = select_checkpoint(scores, lambda path: scores[path])

    assert selection.selected.path == Path("data/v1.pt")
    assert [candidate.path for candidate in selection.rejected] == [Path("data/v2.pt")]


def test_can_select_failed_regression_candidate_when_policy_allows_it():
    scores = {
        Path("data/v1.pt"): record("v1", 0.70, True),
        Path("data/v2.pt"): record("v2", 0.95, False),
    }

    selection = select_checkpoint(scores, lambda path: scores[path], require_regression_pass=False)

    assert selection.selected.path == Path("data/v2.pt")
    assert selection.rejected == ()


def test_ties_are_deterministic_by_path():
    scores = {
        Path("data/b.pt"): record("b", 0.80, True),
        Path("data/a.pt"): record("a", 0.80, True),
    }

    selection = select_checkpoint(scores, lambda path: scores[path])

    assert selection.selected.path == Path("data/b.pt")


def test_empty_checkpoint_set_is_rejected():
    with pytest.raises(ValueError, match="at least one checkpoint"):
        select_checkpoint([], lambda path: record("unused", 0.0))


def test_all_failed_candidates_are_rejected():
    scores = {
        Path("data/v1.pt"): record("v1", 0.70, False),
        Path("data/v2.pt"): record("v2", 0.80, False),
    }

    with pytest.raises(ValueError, match="no checkpoint satisfies"):
        select_checkpoint(scores, lambda path: scores[path])
