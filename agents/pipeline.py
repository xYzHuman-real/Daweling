"""Integrated multi-agent pipeline: collaborate, execute, verify, and review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from core.models import Action, Goal, Observation, Plan, VerificationResult
from core.runtime import Runtime
from planner import Planner

from .collaboration import AgentCollaborator, CollaborationResult
from .debate import AgentDebate, DebateResult


ActionBuilder = Callable[[Plan, dict[str, Any]], Iterable[Action]]


@dataclass
class AgentPipelineResult:
    """Inspectable output from the integrated agent execution pipeline."""

    plan: Plan
    collaboration: CollaborationResult
    observations: list[Observation]
    verifications: list[VerificationResult]
    review: DebateResult | None = None

    @property
    def success(self) -> bool:
        """Require execution, verification, and review to succeed when review is enabled."""
        verified = bool(self.verifications) and all(item.valid for item in self.verifications)
        return verified and (self.review is None or self.review.accepted)


class AgentPipeline:
    """Connect specialist collaboration to runtime execution and independent review."""

    def __init__(
        self,
        collaborator: AgentCollaborator,
        runtime: Runtime,
        *,
        planner: Planner | None = None,
        debate: AgentDebate | None = None,
    ) -> None:
        self.collaborator = collaborator
        self.runtime = runtime
        self.planner = planner or Planner()
        self.debate = debate

    def run(self, goal: Goal, action_builder: ActionBuilder) -> AgentPipelineResult:
        plan = self.planner.create_plan(goal)
        collaboration = self.collaborator.collaborate(plan)
        actions = list(action_builder(plan, collaboration.context))
        self._validate_actions(plan, actions)
        observations = self.runtime.execute(plan, actions)
        verifications = [self.runtime.verify(item) for item in observations]

        review = None
        if self.debate is not None and all(item.valid for item in verifications):
            review = self.debate.review(
                {"observations": observations, "verifications": verifications, "agent_work": collaboration.context},
                task=f"Review the completed work for goal: {goal.description}",
                context={"goal": goal.description},
            )

        return AgentPipelineResult(plan, collaboration, observations, verifications, review)

    @staticmethod
    def _validate_actions(plan: Plan, actions: list[Action]) -> None:
        task_ids = {task.id for task in plan.tasks}
        for action in actions:
            if action.task_id not in task_ids:
                raise ValueError(f"Action references unknown task: {action.task_id}")
            if not action.tool.strip():
                raise ValueError(f"Action tool cannot be empty: {action.task_id}")
