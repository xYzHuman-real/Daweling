"""Registry for discovering and retrieving Daweling tools."""

from __future__ import annotations

from typing import Dict, Iterable

from .base import BaseTool


class ToolRegistry:
    """Manage the executable tools available to the Daweling runtime."""

    def __init__(self, tools: Iterable[BaseTool] | None = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its unique name."""
        name = tool.name.strip()
        if not name:
            raise ValueError("Tool name cannot be empty")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Return a registered tool, or None when it does not exist."""
        return self._tools.get(name)

    def require(self, name: str) -> BaseTool:
        """Return a registered tool or raise a clear lookup error."""
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        return tool

    def list(self) -> list[BaseTool]:
        """Return all registered tools in registration order."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return registered tool names in registration order."""
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
