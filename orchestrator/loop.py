"""Bounded decision-driven workflow loop for Daweling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from core.decision import Decision, DecisionContext, DecisionEngine, NextStep
from core.models import Action, Goal, Observation, Plan, VerificationResult
from core.runtime import Runtime
from planner import Planner

ActionBuilder = Callable[[Plan], Iterable[Action]]
RecoveryHandler = Callable[[Plan, Action, Observation], Action | None]
ReplanHandler = Callable[[Plan, list[Observation], list[VerificationResult], list[Action]], Plan]
ReviewHandler = Callable[[Plan, list[Observation], list[VerificationResult]], bool]


@dataclass
class LoopResult:
    """Inspectable result from a bounded decision-driven workflow."""

    plan: Plan
    observations: list[Observation] = field(default_factory=list)
    verifications: list[VerificationResult] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    recovery_attempts: int = 0
    replan_rounds: int = 0
    reviewed: bool = False

    @property
    def success(self) -> bool:
        return bool(self.decisions) and self.decisions[-1].next_step is NextStep.COMPLETE


class DecisionDrivenLoop:
    """Drive recovery, review, and replanning from one bounded policy engine."""

    def __init__(self, runtime: Runtime, *, planner: Planner | None = None,
                 decision_engine: DecisionEngine | None = None, max_recovery_attempts: int = 2,
                 max_replan_rounds: int = 2) -> None:
        if max_recovery_attempts < 0 or max_replan_rounds < 0:
            raise ValueError("workflow budgets cannot be negative")
        self.runtime = runtime
        self.planner = planner or Planner()
        self.decision_engine = decision_engine or DecisionEngine()
        self.max_recovery_attempts = max_recovery_attempts
        self.max_replan_rounds = max_replan_rounds

    def run(self, goal: Goal, action_builder: ActionBuilder, *, initial_plan: Plan | None = None,
            recover: RecoveryHandler | None = None, replan: ReplanHandler | None = None,
            review: ReviewHandler | None = None, review_required: bool = False) -> LoopResult:
        plan = initial_plan or self.planner.create_plan(goal)
        result = LoopResult(plan)
        recovery_attempts = replan_rounds = 0
        root_action_builder = current_action_builder = action_builder

        while True:
            actions = list(current_action_builder(plan))
            self._validate_actions(plan, actions)
            result.observations = self.runtime.execute(plan, actions)
            result.verifications = [self.runtime.verify(item) for item in result.observations]
            decision = self.decision_engine.decide(DecisionContext(
                plan=plan, observations=tuple(result.observations), verifications=tuple(result.verifications),
                recovery_attempts=recovery_attempts, max_recovery_attempts=self.max_recovery_attempts,
                replan_rounds=replan_rounds, max_replan_rounds=self.max_replan_rounds, review_required=review_required,
            ))
            result.decisions.append(decision)

            if decision.next_step in (NextStep.COMPLETE, NextStep.FAIL):
                result.recovery_attempts, result.replan_rounds = recovery_attempts, replan_rounds
                return result
            if decision.next_step is NextStep.REVIEW:
                if review is None or not review(plan, result.observations, result.verifications):
                    result.decisions.append(Decision(NextStep.FAIL, "Required peer review was unavailable or rejected."))
                    result.reviewed = True
                    return result
                result.reviewed, review_required = True, False
                continue
            if decision.next_step is NextStep.RECOVER:
                if recover is None:
                    result.decisions.append(Decision(NextStep.FAIL, "Recovery was selected but no recovery handler is configured."))
                    return result
                failed_index = next((i for i, item in enumerate(result.verifications) if not item.valid), None)
                if failed_index is None:
                    result.decisions.append(Decision(NextStep.FAIL, "Recovery was selected without a failed verification."))
                    return result
                replacement = recover(plan, actions[failed_index], result.observations[failed_index])
                recovery_attempts += 1
                if replacement is None:
                    result.decisions.append(Decision(NextStep.FAIL, "Recovery handler could not produce a replacement action."))
                    return result
                previous_builder = current_action_builder
                current_action_builder = lambda current_plan, previous=previous_builder, replacement=replacement: [
                    replacement if action.task_id == replacement.task_id else action for action in previous(current_plan)
                ]
                continue
            if decision.next_step is NextStep.REPLAN:
                if replan is None:
                    result.decisions.append(Decision(NextStep.FAIL, "Replanning was selected but no replan handler is configured."))
                    return result
                plan = replan(plan, result.observations, result.verifications, actions)
                result.plan = plan
                replan_rounds += 1
                recovery_attempts = 0
                current_action_builder = root_action_builder

    @staticmethod
    def _validate_actions(plan: Plan, actions: list[Action]) -> None:
        task_ids = {task.id for task in plan.tasks}
        for action in actions:
            if action.task_id not in task_ids:
                raise ValueError(f"Action references unknown task: {action.task_id}")
            if not action.tool.strip():
                raise ValueError(f"Action tool cannot be empty: {action.task_id}")
