"""Minimal execution runtime for Daweling's first core loop."""

from typing import Any, Callable, Dict, Iterable

from .models import Action, Goal, Observation, Plan, Task, VerificationResult, WorkflowState


Tool = Callable[[Dict[str, Any]], Any]


class Runtime:
    """Execute a plan through registered tools and verify each observation."""

    def __init__(self, tools: Dict[str, Tool] | None = None) -> None:
        self.tools: Dict[str, Tool] = tools or {}

    def register_tool(self, name: str, tool: Tool) -> None:
        if not name.strip():
            raise ValueError("Tool name cannot be empty")
        self.tools[name] = tool

    def execute(self, plan: Plan, actions: Iterable[Action]) -> list[Observation]:
        observations: list[Observation] = []

        for action in actions:
            task = self._find_task(plan, action.task_id)
            task.status = WorkflowState.RUNNING

            tool = self.tools.get(action.tool)
            if tool is None:
                observation = Observation(
                    task_id=action.task_id,
                    success=False,
                    error=f"Unknown tool: {action.tool}",
                )
            else:
                try:
                    output = tool(action.input)
                    observation = Observation(
                        task_id=action.task_id,
                        success=True,
                        output=output,
                    )
                except Exception as exc:
                    observation = Observation(
                        task_id=action.task_id,
                        success=False,
                        error=str(exc),
                    )

            observations.append(observation)
            task.status = (
                WorkflowState.COMPLETED
                if observation.success
                else WorkflowState.FAILED
            )

        return observations

    def verify(self, observation: Observation) -> VerificationResult:
        if observation.success:
            return VerificationResult.passed("Tool execution completed successfully")
        return VerificationResult.failed(observation.error or "Tool execution failed")

    @staticmethod
    def _find_task(plan: Plan, task_id: str) -> Task:
        for task in plan.tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"Unknown task: {task_id}")
