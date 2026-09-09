"""Execution runtime with policy checks for Daweling's core loop."""

from typing import Iterable

from tools.base import ToolResult
from tools.registry import ToolRegistry

from .models import Action, Observation, Plan, Task, VerificationResult, WorkflowState
from .policy import ApprovalPolicy, ToolRisk


class Runtime:
    """Execute a plan through registered tools and enforce execution policy."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.approval_policy = approval_policy or ApprovalPolicy()

    def register_tool(self, tool) -> None:
        """Register a BaseTool with this runtime."""
        self.registry.register(tool)

    def execute(self, plan: Plan, actions: Iterable[Action]) -> list[Observation]:
        observations: list[Observation] = []

        for action in actions:
            task = self._find_task(plan, action.task_id)
            task.status = WorkflowState.RUNNING
            tool = self.registry.get(action.tool)

            if tool is None:
                observation = Observation(
                    task_id=action.task_id,
                    success=False,
                    error=f"Unknown tool: {action.tool}",
                )
            elif tool.risk_level == ToolRisk.APPROVAL_REQUIRED and not self.approval_policy.allows(
                tool.name,
                f"Execute '{tool.name}' for task '{task.description}'",
            ):
                observation = Observation(
                    task_id=action.task_id,
                    success=False,
                    error=f"Approval required before executing tool: {tool.name}",
                )
            else:
                try:
                    result = tool(action.input)
                    if not isinstance(result, ToolResult):
                        result = ToolResult.ok(result)
                    observation = Observation(
                        task_id=action.task_id,
                        success=result.success,
                        output=result.output,
                        error=result.error,
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
