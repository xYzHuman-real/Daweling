"""Structured reasoning contracts for Daweling's intelligence core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from models import ModelMessage, ModelProvider


@dataclass(frozen=True)
class ReasoningStep:
    """One concise, auditable reasoning step."""

    purpose: str
    conclusion: str


@dataclass(frozen=True)
class ReasoningResult:
    """Structured reasoning state produced before planning or action."""

    understanding: str
    strategy: str
    steps: tuple[ReasoningStep, ...]
    uncertainties: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "understanding": self.understanding,
            "strategy": self.strategy,
            "steps": [{"purpose": step.purpose, "conclusion": step.conclusion} for step in self.steps],
            "uncertainties": list(self.uncertainties),
        }


class ReasoningEngine:
    """Turn a goal and context into bounded structured reasoning evidence."""

    def __init__(self, provider: ModelProvider, *, max_steps: int = 6, max_uncertainties: int = 4) -> None:
        if max_steps < 1 or max_uncertainties < 0:
            raise ValueError("reasoning limits are invalid")
        self.provider = provider
        self.max_steps = max_steps
        self.max_uncertainties = max_uncertainties

    def reason(self, goal: str, context: dict[str, Any] | None = None) -> ReasoningResult:
        if not goal.strip():
            raise ValueError("Goal cannot be empty")
        response = self.provider.generate([
            ModelMessage(
                role="system",
                content=(
                    "You are Daweling's structured reasoning engine. Return JSON only with keys "
                    "understanding, strategy, steps, uncertainties. Keep reasoning concise and auditable. "
                    "Do not reveal private chain-of-thought. Summarize useful conclusions instead. "
                    "steps must be an ordered list of {purpose, conclusion}. Do not execute tools or claim "
                    "facts not supported by the supplied context."
                ),
            ),
            ModelMessage(
                role="user",
                content=json.dumps({"goal": goal, "context": context or {}}, ensure_ascii=False),
            ),
        ])
        return self._parse(response.content)

    def _parse(self, content: str) -> ReasoningResult:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Reasoning engine returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Reasoning engine response must be an object")

        understanding = str(data.get("understanding", "")).strip()
        strategy = str(data.get("strategy", "")).strip()
        raw_steps = data.get("steps")
        raw_uncertainties = data.get("uncertainties", [])
        if not understanding or not strategy or not isinstance(raw_steps, list):
            raise ValueError("Reasoning engine returned incomplete reasoning")
        if not isinstance(raw_uncertainties, list):
            raise ValueError("Reasoning engine returned invalid uncertainties")
        if len(raw_steps) > self.max_steps or len(raw_uncertainties) > self.max_uncertainties:
            raise ValueError("Reasoning engine exceeded configured bounds")

        steps: list[ReasoningStep] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                raise ValueError("Reasoning step must be an object")
            purpose = str(item.get("purpose", "")).strip()
            conclusion = str(item.get("conclusion", "")).strip()
            if not purpose or not conclusion:
                raise ValueError("Reasoning step is incomplete")
            steps.append(ReasoningStep(purpose, conclusion))
        if not steps:
            raise ValueError("Reasoning engine returned no steps")

        uncertainties = tuple(str(item).strip() for item in raw_uncertainties)
        if any(not item for item in uncertainties):
            raise ValueError("Reasoning uncertainty cannot be empty")
        return ReasoningResult(understanding, strategy, tuple(steps), uncertainties)
