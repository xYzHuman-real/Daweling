"""Context retrieval and formatting for Daweling planning."""

from dataclasses import dataclass
from typing import Any

from core.models import Goal

from .store import MemoryEntry, MemoryStore


@dataclass(frozen=True)
class ContextBundle:
    """Relevant memories packaged for a workflow step."""

    goal: Goal
    memories: list[MemoryEntry]

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal.description,
            "goal_context": self.goal.context,
            "memories": [
                {
                    "key": entry.key,
                    "value": entry.value,
                    "category": entry.category,
                    "created_at": entry.created_at,
                }
                for entry in self.memories
            ],
        }

    def as_prompt_context(self) -> str:
        if not self.memories:
            return "No relevant stored memory was found."
        lines = ["Relevant stored memory:"]
        for entry in self.memories:
            lines.append(f"- [{entry.category}] {entry.key}: {entry.value}")
        return "\n".join(lines)


class ContextEngine:
    """Retrieve relevant memory without exposing the storage implementation."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def build(self, goal: Goal) -> ContextBundle:
        memories = self.memory.search(goal.description)
        return ContextBundle(goal=goal, memories=memories)
