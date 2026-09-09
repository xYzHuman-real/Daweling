"""Structured workflow experience recording for Daweling."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.models import VerificationResult

from .store import MemoryStore


@dataclass(frozen=True)
class WorkflowExperience:
    """A compact, durable summary of one workflow outcome."""

    goal: str
    success: bool
    verification_reason: str
    task_count: int
    successful_tasks: int
    failed_tasks: int
    tools_used: list[str] = field(default_factory=list)
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_value(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "success": self.success,
            "verification_reason": self.verification_reason,
            "task_count": self.task_count,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "tools_used": self.tools_used,
            "recorded_at": self.recorded_at,
        }


class ExperienceRecorder:
    """Persist workflow outcomes without storing raw tool payloads."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def record(
        self,
        goal: str,
        success: bool,
        verifications: list[VerificationResult],
        task_count: int,
        successful_tasks: int,
        failed_tasks: int,
        tools_used: list[str],
    ) -> WorkflowExperience:
        """Record a compact workflow summary as durable memory."""
        reasons = [result.reason for result in verifications if result.reason]
        reason = "; ".join(reasons) or "No verification details recorded"
        experience = WorkflowExperience(
            goal=goal.strip(),
            success=success,
            verification_reason=reason,
            task_count=task_count,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            tools_used=list(dict.fromkeys(tools_used)),
        )
        key = f"experience:{experience.recorded_at}"
        self.memory.remember(key, experience.as_value(), category="experience")
        return experience
