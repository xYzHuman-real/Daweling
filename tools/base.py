"""Base abstractions for Daweling tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned by every Daweling tool."""

    success: bool
    output: Any = None
    error: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: Any = None, **metadata: Any) -> "ToolResult":
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> "ToolResult":
        return cls(success=False, error=error, metadata=metadata)


class BaseTool(ABC):
    """Contract implemented by executable Daweling tools."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute the tool using validated input data."""
        raise NotImplementedError

    def __call__(self, input_data: Dict[str, Any]) -> ToolResult:
        return self.run(input_data)
