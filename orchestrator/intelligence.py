"""High-level intelligence coordination for Daweling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.models import Goal, Plan
from core.reasoning import ReasoningEngine, ReasoningResult
from memory.context import ContextBundle
from planner.model_planner import ModelPlanner


@dataclass(frozen=True)
class IntelligenceResult:
    """Reasoning evidence paired with the resulting execution plan."""

    reasoning: ReasoningResult
    plan: Plan

    def as_dict(self) -> dict[str, Any]:
        return {"reasoning": self.reasoning.as_dict(), "plan": {"tasks": [{"id": t.id, "description": t.description} for t in self.plan.tasks]}}


class IntelligenceCore:
    """Run structured reasoning before model-backed planning."""

    def __init__(self, reasoning: ReasoningEngine, planner: ModelPlanner) -> None:
        self.reasoning = reasoning
        self.planner = planner

    def prepare(self, goal: Goal, context: ContextBundle | None = None) -> IntelligenceResult:
        reasoning_context = context.as_dict() if context is not None else goal.context
        reasoning = self.reasoning.reason(goal.description, reasoning_context)
        plan_context: dict[str, Any] = {
            "reasoning": reasoning.as_dict(),
            "goal_context": goal.context,
        }
        if context is not None:
            plan_context.update(context.as_dict())
        enriched_goal = Goal(goal.description, plan_context)
        plan = self.planner.create_plan(enriched_goal, context=context)
        return IntelligenceResult(reasoning=reasoning, plan=plan)
