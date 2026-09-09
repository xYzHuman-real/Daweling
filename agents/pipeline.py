"""Integrated multi-agent pipeline: collaborate, execute, verify, and review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from core.decision import Decision, DecisionContext, DecisionEngine, NextStep
from core.models import Action, Goal, Observation, Plan, VerificationResult
from core.runtime import Runtime
from planner import Planner

from .collaboration import AgentCollaborator, CollaborationResult
from .debate import AgentDebate, DebateResult


ActionBuilder = Callable[[Plan, dict[str, Any]], Iterable[Action]]


@dataclass
class AgentPipelineResult:
    """Inspectable output from the integrated multi-agent execution pipeline."""

    plan: Plan
    collaboration: CollaborationResult
    observations: list[Observation]
    verifications: list[VerificationResult]
    review: DebateResult | None = None
    decision: Decision | None = None

    @property
    def success(self) -> bool:
        """Return True only when the decision engine reaches COMPLETE."""
        return self.decision is not None and self.decision.next_step is NextStep.COMPLETE


class AgentPipeline:
    """Connect collaboration, execution, verification, review, and control decisions."""

    def __init__(
        self,
        collaborator: AgentCollaborator,
        runtime: Runtime,
        *,
        planner: Planner | None = None,
        debate: AgentDebate | None = None,
        decision_engine: DecisionEngine | None = None,
        review_required: bool | None = None,
    ) -> None:
        self.collaborator = collaborator
        self.runtime = runtime
        self.planner = planner or Planner()
        self.debate = debate
        self.decision_engine = decision_engine or DecisionEngine()
        self.review_required = debate is not None if review_required is None else review_required

    def run(self, goal: Goal, action_builder: ActionBuilder) -> AgentPipelineResult:
        plan = self.planner.create_plan(goal)
        collaboration = self.collaborator.collaborate(plan)
        actions = list(action_builder(plan, collaboration.context))
        self._validate_actions(plan, actions)
        observations = self.runtime.execute(plan, actions)
        verifications = [self.runtime.verify(item) for item in observations]

        context = DecisionContext(
            plan=plan,
            observations=tuple(observations),
            verifications=tuple(verifications),
            collaboration=collaboration,
            review_required=self.review_required,
        )
        decision = self.decision_engine.decide(context)

        review = None
        if decision.next_step is NextStep.REVIEW and self.debate is not None:
            review = self.debate.review(
                {"observations": observations, "verifications": verifications, "agent_work": collaboration.context},
                task=f"Review the completed work for goal: {goal.description}",
                context={"goal": goal.description},
            )
            context = DecisionContext(
                plan=plan,
                observations=tuple(observations),
                verifications=tuple(verifications),
                collaboration=collaboration,
                review=review,
                review_required=self.review_required,
            )
            decision = self.decision_engine.decide(context)
        elif decision.next_step is NextStep.REVIEW:
            decision = Decision(NextStep.FAIL, "Peer review is required but no review engine is configured.")

        return AgentPipelineResult(plan, collaboration, observations, verifications, review, decision)

    @staticmethod
    def _validate_actions(plan: Plan, actions: list[Action]) -> None:
        task_ids = {task.id for task in plan.tasks}
        for action in actions:
            if action.task_id not in task_ids:
                raise ValueError(f"Action references unknown task: {action.task_id}")
            if not action.tool.strip():
                raise ValueError(f"Action tool cannot be empty: {action.task_id}")
