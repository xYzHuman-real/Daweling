"""Application-level orchestration for Daweling's core execution loop."""

from dataclasses import dataclass
from typing import Callable, Iterable

from core.models import Action, Goal, Observation, Plan, VerificationResult
from core.runtime import Runtime
from planner import Planner


ActionBuilder = Callable[[Plan], Iterable[Action]]


@dataclass
class ExecutionResult:
    """Complete result of one Daweling goal execution."""

    plan: Plan
    observations: list[Observation]
    verifications: list[VerificationResult]

    @property
    def success(self) -> bool:
        """Return True only when every executed observation verifies successfully."""
        return bool(self.verifications) and all(result.valid for result in self.verifications)


class Orchestrator:
    """Connect planning, action generation, execution, and verification."""

    def __init__(
        self,
        planner: Planner | None = None,
        runtime: Runtime | None = None,
    ) -> None:
        self.planner = planner or Planner()
        self.runtime = runtime or Runtime()

    def run(self, goal: Goal, action_builder: ActionBuilder) -> ExecutionResult:
        """Run a goal through Plan → Action → Execute → Verify."""
        plan = self.planner.create_plan(goal)
        actions = list(action_builder(plan))
        self._validate_actions(plan, actions)

        observations = self.runtime.execute(plan, actions)
        verifications = [self.runtime.verify(observation) for observation in observations]

        return ExecutionResult(
            plan=plan,
            observations=observations,
            verifications=verifications,
        )

    @staticmethod
    def _validate_actions(plan: Plan, actions: list[Action]) -> None:
        task_ids = {task.id for task in plan.tasks}
        for action in actions:
            if action.task_id not in task_ids:
                raise ValueError(f"Action references unknown task: {action.task_id}")
            if not action.tool.strip():
                raise ValueError(f"Action tool cannot be empty: {action.task_id}")
