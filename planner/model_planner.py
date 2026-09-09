"""Model-backed planning adapter for Daweling.

The adapter deliberately keeps model output separate from domain models so a
provider can be replaced without changing the rest of the planning system.
"""

import json
from typing import Any

from core.models import Goal, Plan, Task
from memory.context import ContextBundle
from memory.context_formatter import build_context_message
from models import ModelMessage, ModelProvider


class ModelPlanner:
    """Use a ModelProvider to turn a goal and relevant memory into a task plan."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def create_plan(self, goal: Goal, context: ContextBundle | None = None) -> Plan:
        if not goal.description.strip():
            raise ValueError("Goal description cannot be empty")

        messages = [
            ModelMessage(
                role="system",
                content=(
                    "You are Daweling's planning engine. Return JSON only: "
                    "{\"tasks\":[{\"id\":\"...\",\"description\":\"...\"}]} . "
                    "Create a minimal ordered task list. Do not execute actions."
                ),
            )
        ]
        if context is not None:
            messages.append(build_context_message(context))
        messages.append(
            ModelMessage(
                role="user",
                content=json.dumps(
                    {"goal": goal.description, "context": goal.context},
                    ensure_ascii=False,
                ),
            )
        )

        response = self.provider.generate(messages)
        data = self._parse_json(response.content)
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("Model planner returned no tasks")

        tasks = [self._task_from_item(item, index) for index, item in enumerate(raw_tasks, 1)]
        self._validate_unique_ids(tasks)
        return Plan(goal=goal, tasks=tasks)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Model planner returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Model planner response must be a JSON object")
        return data

    @staticmethod
    def _task_from_item(item: Any, index: int) -> Task:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid task at index {index}")
        task_id = str(item.get("id", f"task-{index}")).strip()
        description = str(item.get("description", "")).strip()
        if not task_id or not description:
            raise ValueError(f"Invalid task at index {index}")
        return Task(id=task_id, description=description)

    @staticmethod
    def _validate_unique_ids(tasks: list[Task]) -> None:
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Model planner returned duplicate task ids")
