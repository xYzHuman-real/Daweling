"""Bounded multi-agent collaboration with shared work products."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.models import Plan

from .base import AgentResult
from .router import AgentRouter


@dataclass(frozen=True)
class CollaborationStep:
    """One specialist contribution to a shared workflow."""

    task_id: str
    agent: str
    result: AgentResult


@dataclass
class CollaborationResult:
    """All specialist contributions and the shared context they produced."""

    steps: list[CollaborationStep] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.steps) and all(step.result.success for step in self.steps)

    @property
    def context(self) -> dict[str, Any]:
        return {
            step.task_id: {
                "agent": step.agent,
                "success": step.result.success,
                "output": step.result.output,
                "error": step.result.error,
                "metadata": step.result.metadata,
            }
            for step in self.steps
        }


class AgentCollaborator:
    """Run relevant agents in plan order, feeding prior work into later agents."""

    def __init__(self, router: AgentRouter, *, max_context_items: int = 8) -> None:
        if max_context_items < 1:
            raise ValueError("max_context_items must be positive")
        self.router = router
        self.max_context_items = max_context_items

    def collaborate(
        self,
        plan: Plan,
        context: dict[str, Any] | None = None,
    ) -> CollaborationResult:
        shared: dict[str, Any] = dict(context or {})
        result = CollaborationResult()
        for task in plan.tasks:
            agent = self.router.route(task)
            if agent is None:
                continue
            bounded_context = dict(list(shared.items())[-self.max_context_items :])
            agent_result = agent.run(task.description, context=bounded_context)
            step = CollaborationStep(task.id, agent.name, agent_result)
            result.steps.append(step)
            shared[task.id] = {
                "agent": agent.name,
                "success": agent_result.success,
                "output": agent_result.output,
                "error": agent_result.error,
            }
        return result
