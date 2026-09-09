"""Context retrieval and formatting for Daweling planning."""

from dataclasses import dataclass
from typing import Any

from core.models import Goal
from .learning import LearningContext, LearningEngine, PlanningGuidance
from .store import MemoryEntry, MemoryStore


@dataclass(frozen=True)
class ContextBundle:
    """Relevant project memories and learned workflow guidance for a goal."""

    goal: Goal
    memories: list[MemoryEntry]
    learning: LearningContext

    @property
    def guidance(self) -> PlanningGuidance:
        return self.learning.guidance()

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal.description,
            "goal_context": self.goal.context,
            "memories": [
                {"key": e.key, "value": e.value, "category": e.category, "created_at": e.created_at}
                for e in self.memories
            ],
            "experiences": [e.value for e in self.learning.experiences],
            "guidance": self.guidance.as_dict(),
        }

    def as_prompt_context(self) -> str:
        if self.memories:
            memory = "\n".join(f"- [{e.category}] {e.key}: {e.value}" for e in self.memories)
            memory_section = f"Relevant stored memory:\n{memory}"
        else:
            memory_section = "No relevant stored memory was found."
        return "\n\n".join((memory_section, self.learning.as_prompt_context(), self.guidance.as_prompt_context()))


class ContextEngine:
    """Retrieve project memory, prior experience, and reusable strategy guidance."""

    _STOPWORDS = frozenset({"a", "an", "and", "build", "create", "for", "from", "in", "of", "on", "the", "to", "with"})

    def __init__(self, memory: MemoryStore, learning: LearningEngine | None = None) -> None:
        self.memory = memory
        self.learning = learning or LearningEngine(memory)

    def build(self, goal: Goal) -> ContextBundle:
        terms = [t for t in goal.description.split() if t.casefold() not in self._STOPWORDS]
        query = " ".join(terms) or goal.description
        memories = [e for e in self.memory.search(query) if e.category != "experience"]
        return ContextBundle(goal=goal, memories=memories, learning=self.learning.build(goal.description))
