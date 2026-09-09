"""Deterministic workflow decisions for Daweling orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from agents.collaboration import CollaborationResult
from agents.debate import DebateResult
from core.models import Observation, Plan, VerificationResult


class NextStep(str, Enum):
    """The next bounded control-plane transition for a workflow."""

    EXECUTE = "execute"
    REVIEW = "review"
    RECOVER = "recover"
    REPLAN = "replan"
    COMPLETE = "complete"
    FAIL = "fail"


@dataclass(frozen=True)
class DecisionContext:
    """Evidence used by the decision engine; no model call is required."""

    plan: Plan
    observations: tuple[Observation, ...] = ()
    verifications: tuple[VerificationResult, ...] = ()
    collaboration: CollaborationResult | None = None
    review: DebateResult | None = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 2
    replan_rounds: int = 0
    max_replan_rounds: int = 2
    review_required: bool = False
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class Decision:
    """An auditable next-step decision and the evidence-based reason."""

    next_step: NextStep
    reason: str


class DecisionEngine:
    """Choose workflow transitions with deterministic, bounded policy."""

    def decide(self, context: DecisionContext) -> Decision:
        if context.recovery_attempts < 0 or context.replan_rounds < 0:
            raise ValueError("workflow counters cannot be negative")
        if context.max_recovery_attempts < 0 or context.max_replan_rounds < 0:
            raise ValueError("workflow budgets cannot be negative")

        if not context.observations:
            return Decision(NextStep.EXECUTE, "No execution evidence exists yet.")

        if not context.verifications or len(context.verifications) != len(context.observations):
            return Decision(NextStep.FAIL, "Execution evidence is incomplete and cannot be verified safely.")

        if not all(item.valid for item in context.verifications):
            if context.recovery_attempts < context.max_recovery_attempts:
                return Decision(NextStep.RECOVER, "Verification failed and bounded recovery attempts remain.")
            if context.replan_rounds < context.max_replan_rounds:
                return Decision(NextStep.REPLAN, "Verification failed and recovery budget is exhausted; replan budget remains.")
            return Decision(NextStep.FAIL, "Verification failed and all recovery/replan budgets are exhausted.")

        if context.review_required and context.review is None:
            return Decision(NextStep.REVIEW, "Verification passed and required peer review has not run.")

        if context.review is not None and not context.review.accepted:
            if context.replan_rounds < context.max_replan_rounds:
                return Decision(NextStep.REPLAN, "Peer review rejected the work and replan budget remains.")
            return Decision(NextStep.FAIL, "Peer review rejected the work and no replan budget remains.")

        return Decision(NextStep.COMPLETE, "All available verification and required review gates passed.")
