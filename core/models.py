"""Small, dependency-free domain models for the Daweling runtime."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class WorkflowState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Goal:
    description: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    id: str
    description: str
    status: WorkflowState = WorkflowState.PENDING


@dataclass
class Plan:
    goal: Goal
    tasks: List[Task] = field(default_factory=list)


@dataclass
class Action:
    task_id: str
    tool: str
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    task_id: str
    success: bool
    output: Any = None
    error: str | None = None


@dataclass
class VerificationResult:
    valid: bool
    reason: str

    @classmethod
    def passed(cls, reason: str = "Verification passed") -> "VerificationResult":
        return cls(valid=True, reason=reason)

    @classmethod
    def failed(cls, reason: str) -> "VerificationResult":
        return cls(valid=False, reason=reason)
