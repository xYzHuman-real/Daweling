"""Base abstractions for specialized Daweling agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentResult:
    """Normalized result returned by an agent."""

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: Any = None, **metadata: Any) -> "AgentResult":
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> "AgentResult":
        return cls(success=False, error=error, metadata=metadata)


class BaseAgent(ABC):
    """Contract for a specialized agent capability."""

    name: str = ""
    description: str = ""
    capabilities: tuple[str, ...] = ()

    @abstractmethod
    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Execute an agent task using supplied context."""
        raise NotImplementedError

    def __call__(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        return self.run(task, context)
