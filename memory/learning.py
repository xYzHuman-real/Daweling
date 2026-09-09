"""Learning-oriented retrieval over recorded Daweling experiences."""

from dataclasses import dataclass
from typing import Any

from .store import MemoryEntry, MemoryStore


@dataclass(frozen=True)
class LearningContext:
    """Relevant past experiences selected for a new goal."""

    experiences: list[MemoryEntry]

    def as_prompt_context(self) -> str:
        if not self.experiences:
            return "No relevant past workflow experience was found."

        lines = ["Relevant past workflow experience:"]
        for entry in self.experiences:
            value = entry.value if isinstance(entry.value, dict) else {"summary": entry.value}
            lines.append(
                "- goal={goal!r}; success={success}; tools={tools}; "
                "successful_tasks={successful_tasks}; failed_tasks={failed_tasks}; "
                "verification={verification_reason}".format(
                    goal=value.get("goal", ""),
                    success=value.get("success", "unknown"),
                    tools=value.get("tools_used", []),
                    successful_tasks=value.get("successful_tasks", 0),
                    failed_tasks=value.get("failed_tasks", 0),
                    verification_reason=value.get("verification_reason", ""),
                )
            )
        return "\n".join(lines)


class LearningEngine:
    """Retrieve prior experiences relevant to a new goal."""

    def __init__(self, memory: MemoryStore, max_results: int = 5) -> None:
        if max_results <= 0:
            raise ValueError("max_results must be greater than zero")
        self.memory = memory
        self.max_results = max_results

    def build(self, goal: str) -> LearningContext:
        matches = self.memory.search(goal, category="experience")
        return LearningContext(experiences=matches[: self.max_results])

    def as_dict(self, goal: str) -> dict[str, Any]:
        """Return a structured learning payload for APIs and logging."""
        context = self.build(goal)
        return {
            "experiences": [entry.value for entry in context.experiences],
        }
