"""Deterministic agent router for Daweling's first multi-agent layer."""

from core.models import Task

from .base import BaseAgent
from .registry import AgentRegistry


class AgentRouter:
    """Select the most relevant registered agent from task capabilities."""

    _KEYWORDS = {
        "research": ("research", "investigate", "compare", "sources", "evidence"),
        "coding": ("code", "coding", "debug", "bug", "implement", "software", "program"),
    }

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def route(self, task: Task) -> BaseAgent | None:
        text = task.description.lower()
        scored: list[tuple[int, str, BaseAgent]] = []
        for agent in self.registry.list():
            score = sum(1 for keyword in self._KEYWORDS.get(agent.name, ()) if keyword in text)
            score += sum(1 for capability in agent.capabilities if capability.lower() in text)
            if score:
                scored.append((score, agent.name, agent))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]
