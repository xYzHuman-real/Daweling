"""Execution policy and approval primitives for Daweling."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ToolRisk(str, Enum):
    """Risk classification used by the execution control layer."""

    SAFE = "safe"
    APPROVAL_REQUIRED = "approval_required"


ApprovalCallback = Callable[[str, str], bool]


@dataclass(frozen=True)
class ApprovalPolicy:
    """Decide whether a tool action may execute."""

    approval_callback: ApprovalCallback | None = None

    def allows(self, tool_name: str, reason: str) -> bool:
        """Return True when the action is approved by policy."""
        if self.approval_callback is None:
            return False
        return bool(self.approval_callback(tool_name, reason))
