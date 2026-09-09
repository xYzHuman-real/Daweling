"""Deterministic tool selection for turning planned tasks into actions."""

import re
from typing import Dict

from core.models import Action, Plan
from tools.registry import ToolRegistry


class ActionBuilder:
    """Choose the best registered tool for each task using capability matching.

    This is deliberately model-free for now. A future model-backed selector can
    implement the same interface while retaining this safe deterministic fallback.
    """

    def __init__(self, registry: ToolRegistry, explicit_tools: Dict[str, str] | None = None) -> None:
        self.registry = registry
        self.explicit_tools = explicit_tools or {}

    def build(self, plan: Plan) -> list[Action]:
        """Generate one executable action per task."""
        actions: list[Action] = []
        for task in plan.tasks:
            tool = self._select_tool(task.id, task.description)
            actions.append(
                Action(
                    task_id=task.id,
                    tool=tool.name,
                    input={"task": task.description, "goal": plan.goal.description},
                )
            )
        return actions

    def _select_tool(self, task_id: str, description: str):
        if task_id in self.explicit_tools:
            return self.registry.require(self.explicit_tools[task_id])

        tools = self.registry.list()
        if not tools:
            raise ValueError("Cannot build actions without registered tools")

        task_tokens = self._tokens(description)
        scored = []
        for index, tool in enumerate(tools):
            capability_tokens = self._tokens(f"{tool.name} {tool.description}")
            score = len(task_tokens & capability_tokens)
            scored.append((score, -index, tool))

        best_score, _, best_tool = max(scored, key=lambda item: (item[0], item[1]))
        if best_score == 0 and len(tools) > 1:
            raise ValueError(f"No suitable tool found for task: {task_id}")
        return best_tool

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}
