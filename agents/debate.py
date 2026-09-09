"""Bounded peer review for important multi-agent work products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import AgentResult, BaseAgent


@dataclass(frozen=True)
class ReviewDecision:
    """A reviewer's structured assessment of another agent's work."""

    reviewer: str
    approved: bool
    critique: str
    suggestions: tuple[str, ...] = ()


@dataclass(frozen=True)
class DebateResult:
    """Final decision after bounded independent review."""

    work: Any
    reviews: tuple[ReviewDecision, ...]
    accepted: bool


class AgentDebate:
    """Have independent reviewers challenge a work product before acceptance."""

    def __init__(self, reviewers: list[BaseAgent], *, max_reviewers: int = 2) -> None:
        if max_reviewers < 1:
            raise ValueError("max_reviewers must be positive")
        if not reviewers:
            raise ValueError("At least one reviewer is required")
        self.reviewers = reviewers[:max_reviewers]

    def review(self, work: Any, *, task: str, context: dict[str, Any] | None = None) -> DebateResult:
        reviews: list[ReviewDecision] = []
        for reviewer in self.reviewers:
            review_context = dict(context or {})
            review_context.update({"work_to_review": work, "review_role": "challenge correctness and identify weaknesses"})
            result = reviewer.run(task, context=review_context)
            reviews.append(self._decision(reviewer.name, result))
        return DebateResult(work=work, reviews=tuple(reviews), accepted=bool(reviews) and all(r.approved for r in reviews))

    @staticmethod
    def _decision(name: str, result: AgentResult) -> ReviewDecision:
        if not result.success:
            return ReviewDecision(name, False, result.error or "Reviewer failed")
        output = result.output
        if isinstance(output, dict):
            approved = bool(output.get("approved", False))
            critique = str(output.get("critique", "")).strip() or "No critique provided."
            raw = output.get("suggestions", ())
            suggestions = tuple(str(item) for item in raw) if isinstance(raw, (list, tuple)) else ()
            return ReviewDecision(name, approved, critique, suggestions)
        return ReviewDecision(name, False, "Reviewer must return structured approval evidence.")
