"""Learning-oriented retrieval and strategy guidance for Daweling planning."""

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
                "verification={verification_reason}; recovery_diagnoses={diagnoses}".format(
                    goal=value.get("goal", ""),
                    success=value.get("success", "unknown"),
                    tools=value.get("tools_used", []),
                    successful_tasks=value.get("successful_tasks", 0),
                    failed_tasks=value.get("failed_tasks", 0),
                    verification_reason=value.get("verification_reason", ""),
                    diagnoses=value.get("recovery_diagnoses", []),
                )
            )
        return "\n".join(lines)

    def guidance(self) -> "PlanningGuidance":
        """Derive bounded, auditable planning guidance from prior outcomes."""
        successful_tools: list[str] = []
        failed_tools: list[str] = []
        lessons: list[str] = []
        for entry in self.experiences:
            value = entry.value if isinstance(entry.value, dict) else {}
            tools = [str(tool) for tool in value.get("tools_used", [])]
            if value.get("success") is True:
                successful_tools.extend(tools)
            elif value.get("success") is False:
                failed_tools.extend(tools)
            for diagnosis in value.get("recovery_diagnoses", []):
                text = str(diagnosis).strip()
                if text:
                    lessons.append(text)

        return PlanningGuidance(
            preferred_tools=tuple(dict.fromkeys(successful_tools)),
            avoid_tools=tuple(dict.fromkeys(failed_tools)),
            lessons=tuple(dict.fromkeys(lessons)),
        )


@dataclass(frozen=True)
class PlanningGuidance:
    """Small strategy hints derived from prior workflow evidence."""

    preferred_tools: tuple[str, ...] = ()
    avoid_tools: tuple[str, ...] = ()
    lessons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "preferred_tools": list(self.preferred_tools),
            "avoid_tools": list(self.avoid_tools),
            "lessons": list(self.lessons),
        }

    def as_prompt_context(self) -> str:
        if not (self.preferred_tools or self.avoid_tools or self.lessons):
            return "No reusable strategy guidance was derived from prior workflows."
        return (
            "Reusable strategy guidance:\n"
            f"- preferred_tools={list(self.preferred_tools)}\n"
            f"- avoid_tools={list(self.avoid_tools)}\n"
            f"- lessons={list(self.lessons)}"
        )


class LearningEngine:
    """Retrieve prior experiences and derive reusable planning guidance."""

    def __init__(self, memory: MemoryStore, max_results: int = 5) -> None:
        if max_results <= 0:
            raise ValueError("max_results must be greater than zero")
        self.memory = memory
        self.max_results = max_results

    def build(self, goal: str) -> LearningContext:
        matches = self.memory.search(goal, category="experience")
        return LearningContext(experiences=matches[: self.max_results])

    def as_dict(self, goal: str) -> dict[str, Any]:
        """Return structured learning evidence for APIs and logging."""
        context = self.build(goal)
        return {
            "experiences": [entry.value for entry in context.experiences],
            "guidance": context.guidance().as_dict(),
        }
