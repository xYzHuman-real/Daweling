"""Deterministic first-generation planner for Daweling."""

from typing import Callable

from core.models import Goal, Plan, Task


TaskBuilder = Callable[[Goal], list[Task]]


class Planner:
    """Convert a user goal into a validated, ordered task plan.

    The first implementation is intentionally deterministic. A model-backed
    planner can later implement the same interface without changing callers.
    """

    def __init__(self, task_builder: TaskBuilder | None = None) -> None:
        self._task_builder = task_builder or self._default_task_builder

    def create_plan(self, goal: Goal) -> Plan:
        """Create a plan and reject empty or duplicate task identifiers."""
        if not goal.description.strip():
            raise ValueError("Goal description cannot be empty")

        tasks = self._task_builder(goal)
        self._validate_tasks(tasks)
        return Plan(goal=goal, tasks=tasks)

    @staticmethod
    def _default_task_builder(goal: Goal) -> list[Task]:
        return [Task(id="task-1", description=goal.description.strip())]

    @staticmethod
    def _validate_tasks(tasks: list[Task]) -> None:
        seen: set[str] = set()
        for task in tasks:
            task_id = task.id.strip()
            if not task_id:
                raise ValueError("Task id cannot be empty")
            if not task.description.strip():
                raise ValueError(f"Task description cannot be empty: {task_id}")
            if task_id in seen:
                raise ValueError(f"Duplicate task id: {task_id}")
            seen.add(task_id)
