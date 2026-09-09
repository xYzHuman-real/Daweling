"""Model-backed action generation with strict structural validation."""

import json
from typing import Any

from core.models import Action, Plan
from models import ModelMessage, ModelProvider
from tools.registry import ToolRegistry


class ModelActionBuilder:
    """Ask a model to map planned tasks to registered tools."""

    def __init__(self, provider: ModelProvider, registry: ToolRegistry) -> None:
        self.provider = provider
        self.registry = registry

    def build_actions(self, plan: Plan) -> list[Action]:
        tools = [
            {"name": tool.name, "description": tool.description}
            for tool in self.registry.list()
        ]
        response = self.provider.generate(
            [
                ModelMessage(
                    role="system",
                    content=(
                        "You are Daweling's action planner. Return JSON only: "
                        "{\"actions\":[{\"task_id\":\"...\",\"tool\":\"...\",\"input\":{}}]}. "
                        "Use only the supplied tools and planned task IDs. Never invent tools."
                    ),
                ),
                ModelMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "tasks": [
                                {"id": task.id, "description": task.description}
                                for task in plan.tasks
                            ],
                            "tools": tools,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        return self._parse_actions(response.content, {task.id for task in plan.tasks})

    def _parse_actions(self, content: str, task_ids: set[str]) -> list[Action]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Model action builder returned invalid JSON") from exc

        raw_actions = data.get("actions") if isinstance(data, dict) else None
        if not isinstance(raw_actions, list):
            raise ValueError("Model action builder returned no actions")

        actions: list[Action] = []
        for item in raw_actions:
            if not isinstance(item, dict):
                raise ValueError("Invalid action returned by model")
            task_id = str(item.get("task_id", "")).strip()
            tool = str(item.get("tool", "")).strip()
            input_data = item.get("input", {})
            if task_id not in task_ids:
                raise ValueError(f"Action references unknown task: {task_id}")
            if tool not in self.registry:
                raise ValueError(f"Model selected unavailable tool: {tool}")
            if not isinstance(input_data, dict):
                raise ValueError(f"Action input must be an object: {task_id}")
            actions.append(Action(task_id=task_id, tool=tool, input=input_data))
        return actions
