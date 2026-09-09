"""Context retrieval and formatting for Daweling planning."""

from dataclasses import dataclass
from typing import Any

from core.models import Goal

from .learning import LearningContext, LearningEngine
from .store import MemoryEntry, MemoryStore


@dataclass(frozen=True)
class ContextBundle:
    """Relevant project memories and past experiences for a workflow step."""

    goal: Goal
    memories: list[MemoryEntry]
    learning: LearningContext

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
            "experiences": [entry.value for entry in self.learning.experiences],
        }

    def as_prompt_context(self) -> str:
        sections: list[str] = []
        if self.memories:
            lines = ["Relevant stored memory:"]
            for entry in self.memories:
                lines.append(f"- [{entry.category}] {entry.key}: {entry.value}")
            sections.append("\n".join(lines))
        else:
            sections.append("No relevant stored memory was found.")

        sections.append(self.learning.as_prompt_context())
        return "\n\n".join(sections)


class ContextEngine:
    """Retrieve project memory and prior workflow experience."""

    def __init__(self, memory: MemoryStore, learning: LearningEngine | None = None) -> None:
        self.memory = memory
        self.learning = learning or LearningEngine(memory)

    def build(self, goal: Goal) -> ContextBundle:
        memories = [
            entry
            for entry in self.memory.search(goal.description)
            if entry.category != "experience"
        ]
        learning = self.learning.build(goal.description)
        return ContextBundle(goal=goal, memories=memories, learning=learning)
