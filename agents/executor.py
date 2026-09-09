"""Execution bridge between planned tasks and specialized Daweling agents."""

from dataclasses import dataclass
from typing import Any

from core.models import Plan

from .base import AgentResult
from .router import AgentRouter


@dataclass(frozen=True)
class AgentExecution:
    """Result of routing and executing one planned task through an agent."""

    task_id: str
    agent: str | None
    result: AgentResult


class AgentExecutor:
    """Route every task to a matching agent and collect its output."""

    def __init__(self, router: AgentRouter) -> None:
        self.router = router

    def execute(
        self,
        plan: Plan,
        context: dict[str, Any] | None = None,
    ) -> list[AgentExecution]:
        executions: list[AgentExecution] = []
        for task in plan.tasks:
            agent = self.router.route(task)
            if agent is None:
                continue
            result = agent.run(task.description, context=context)
            executions.append(
                AgentExecution(
                    task_id=task.id,
                    agent=agent.name,
                    result=result,
                )
            )
        return executions

    @staticmethod
    def as_action_context(executions: list[AgentExecution]) -> dict[str, Any]:
        """Convert agent work into bounded, structured context for action planning."""
        return {
            execution.task_id: {
                "agent": execution.agent,
                "success": execution.result.success,
                "output": execution.result.output,
                "error": execution.result.error,
            }
            for execution in executions
        }
