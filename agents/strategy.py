"""Strategy selection for Daweling agent orchestration.

The strategy layer ranks explicit, registered strategies using task text,
agent capabilities, learned guidance, and bounded deterministic scoring.
It does not execute tools or bypass runtime policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from core.models import Task

from .base import BaseAgent
from .registry import AgentRegistry


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    score: float
    rationale: str
    agents: tuple[str, ...]


@dataclass(frozen=True)
class StrategySelection:
    selected: StrategyCandidate
    alternatives: tuple[StrategyCandidate, ...] = ()


@dataclass(frozen=True)
class AgentStrategy:
    name: str
    keywords: tuple[str, ...] = ()
    preferred_capabilities: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()


class StrategySelector:
    """Select a bounded strategy from explicit evidence and available agents."""

    def __init__(self, registry: AgentRegistry, strategies: Iterable[AgentStrategy] | None = None) -> None:
        self.registry = registry
        self.strategies = tuple(strategies or self._default_strategies())
        if not self.strategies:
            raise ValueError("At least one strategy is required")
        names = [strategy.name for strategy in self.strategies]
        if len(names) != len(set(names)):
            raise ValueError("Strategy names must be unique")

    @staticmethod
    def _default_strategies() -> tuple[AgentStrategy, ...]:
        return (
            AgentStrategy("research-first", ("research", "investigate", "compare", "sources", "evidence"), ("research",), ("research",)),
            AgentStrategy("build-first", ("code", "coding", "debug", "implement", "software", "program"), ("coding",), ("coding",)),
            AgentStrategy("mixed", ("analyze", "analysis", "design", "plan", "complex"), ("research", "coding"), ("research", "coding")),
        )

    def select(
        self,
        task: Task,
        *,
        guidance: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_alternatives: int = 2,
    ) -> StrategySelection:
        if max_alternatives < 0:
            raise ValueError("max_alternatives must be non-negative")
        text = task.description.casefold()
        guidance = guidance or {}
        preferred = {str(x).casefold() for x in guidance.get("preferred_tools", [])}
        avoid = {str(x).casefold() for x in guidance.get("avoid_tools", [])}
        available = {agent.name.casefold(): agent for agent in self.registry.list()}

        candidates: list[StrategyCandidate] = []
        for strategy in self.strategies:
            score = 0.0
            reasons: list[str] = []
            keyword_hits = [keyword for keyword in strategy.keywords if keyword.casefold() in text]
            score += min(len(keyword_hits), 4) * 2.0
            if keyword_hits:
                reasons.append(f"task keywords: {', '.join(keyword_hits[:4])}")
            capability_hits = []
            for agent_name in strategy.agents:
                agent = available.get(agent_name.casefold())
                if agent is None:
                    continue
                capability_hits.extend(
                    capability for capability in strategy.preferred_capabilities
                    if any(capability.casefold() == item.casefold() for item in agent.capabilities)
                )
            score += min(len(capability_hits), 4) * 1.5
            if capability_hits:
                reasons.append(f"available capabilities: {', '.join(dict.fromkeys(capability_hits))}")
            for agent_name in strategy.agents:
                name = agent_name.casefold()
                if name in preferred:
                    score += 3.0
                    reasons.append(f"learned preference: {agent_name}")
                if name in avoid:
                    score -= 3.0
                    reasons.append(f"learned warning: avoid {agent_name}")
            selected_agents = tuple(name for name in strategy.agents if name.casefold() in available)
            if not selected_agents:
                score -= 100.0
                reasons.append("required agents unavailable")
            rationale = "; ".join(reasons) or "No strong signal; deterministic baseline strategy."
            candidates.append(StrategyCandidate(strategy.name, score, rationale, selected_agents))

        candidates.sort(key=lambda item: (-item.score, item.name))
        if candidates[0].score < -50:
            raise ValueError("No viable strategy is available for the task")
        return StrategySelection(candidates[0], tuple(candidates[1 : max_alternatives + 1]))


class DynamicAgentRouter:
    """Route tasks through the selected strategy while retaining deterministic fallback behavior."""

    def __init__(self, registry: AgentRegistry, selector: StrategySelector | None = None) -> None:
        self.registry = registry
        self.selector = selector or StrategySelector(registry)

    def route(self, task: Task, *, guidance: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> tuple[BaseAgent, StrategySelection]:
        selection = self.selector.select(task, guidance=guidance, context=context)
        for agent_name in selection.selected.agents:
            agent = self.registry.get(agent_name)
            if agent is not None:
                return agent, selection
        raise ValueError(f"Selected strategy has no available agent: {selection.selected.name}")
