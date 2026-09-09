"""Reasoning-training dataset primitives for Daweling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReasoningExample:
    """A reasoning example with an auditable sequence of steps and final answer."""

    problem: str
    steps: tuple[str, ...]
    answer: str
    category: str = "general"

    def as_instruction_example(self) -> dict[str, str]:
        """Convert to a supervised example without requiring hidden reasoning."""
        reasoning = "\n".join(f"Step {index}: {step}" for index, step in enumerate(self.steps, 1))
        response = f"{reasoning}\nAnswer: {self.answer}" if reasoning else f"Answer: {self.answer}"
        return {"instruction": self.problem, "response": response}


def read_reasoning_examples(path: Path) -> list[ReasoningExample]:
    examples: list[ReasoningExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"line {line_number}: expected an object")
        problem = item.get("problem")
        steps = item.get("steps")
        answer = item.get("answer")
        category = item.get("category", "general")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"line {line_number}: problem must be a non-empty string")
        if not isinstance(steps, list) or not all(isinstance(step, str) and step.strip() for step in steps):
            raise ValueError(f"line {line_number}: steps must be a list of non-empty strings")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"line {line_number}: answer must be a non-empty string")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"line {line_number}: category must be a non-empty string")
        examples.append(ReasoningExample(problem.strip(), tuple(step.strip() for step in steps), answer.strip(), category.strip()))
    if not examples:
        raise ValueError("reasoning dataset is empty")
    return examples


def validate_reasoning_example(example: ReasoningExample) -> tuple[str, ...]:
    """Return validation issues; an empty tuple means the example is valid."""
    issues: list[str] = []
    if not example.problem.strip():
        issues.append("problem is empty")
    if not example.steps:
        issues.append("steps are empty")
    if any(not step.strip() for step in example.steps):
        issues.append("steps must be non-empty")
    if not example.answer.strip():
        issues.append("answer is empty")
    return tuple(issues)
