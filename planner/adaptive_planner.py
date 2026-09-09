"""Adaptive replanning after verified workflow failures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.models import Action, Observation, Plan, Task, VerificationResult
from models import ModelMessage, ModelProvider


@dataclass(frozen=True)
class ReplanResult:
    """A revised plan plus the model's explanation for changing it."""

    plan: Plan
    reason: str


class AdaptivePlanner:
    """Ask the model to revise only the failed portion of a plan using evidence."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def replan(
        self,
        plan: Plan,
        observations: list[Observation],
        verifications: list[VerificationResult],
        actions: list[Action],
    ) -> ReplanResult:
        failures = []
        for action, observation, verification in zip(actions, observations, verifications):
            if not verification.valid:
                failures.append(
                    {
                        "task_id": action.task_id,
                        "task": next(t.description for t in plan.tasks if t.id == action.task_id),
                        "action": {"tool": action.tool, "input": action.input},
                        "observation": {"success": observation.success, "error": observation.error, "output": observation.output},
                        "verification": verification.reason,
                    }
                )
        if not failures:
            return ReplanResult(plan, "No replanning required; all tasks verified successfully.")

        response = self.provider.generate([
            ModelMessage(
                role="system",
                content=(
                    "You are Daweling's adaptive planning engine. Return JSON only: "
                    "{\"reason\":\"...\",\"tasks\":[{\"id\":\"...\",\"description\":\"...\"}]}. "
                    "Use the failure evidence to create a better ordered plan. Preserve successful tasks "
                    "when possible, replace or refine failed tasks, and do not invent facts or tools. "
                    "Task IDs must be unique."
                ),
            ),
            ModelMessage(
                role="user",
                content=json.dumps({
                    "goal": plan.goal.description,
                    "current_plan": [{"id": t.id, "description": t.description} for t in plan.tasks],
                    "failures": failures,
                }, ensure_ascii=False),
            ),
        ])
        data = json.loads(response.content)
        if not isinstance(data, dict) or not isinstance(data.get("tasks"), list) or not data["tasks"]:
            raise ValueError("Adaptive planner returned no tasks")
        tasks: list[Task] = []
        for item in data["tasks"]:
            if not isinstance(item, dict):
                raise ValueError("Adaptive planner returned an invalid task")
            task_id = str(item.get("id", "")).strip()
            description = str(item.get("description", "")).strip()
            if not task_id or not description:
                raise ValueError("Adaptive planner returned an invalid task")
            tasks.append(Task(task_id, description))
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("Adaptive planner returned duplicate task ids")
        reason = str(data.get("reason", "Adaptive replanning completed.")).strip()
        return ReplanResult(Plan(plan.goal, tasks), reason)
