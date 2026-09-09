from agents import AgentRegistry, BaseAgent, AgentResult
from agents.strategy import AgentStrategy, DynamicAgentRouter, StrategySelector
from core.models import Task


class StubAgent(BaseAgent):
    def __init__(self, name, capabilities):
        self.name = name
        self.capabilities = tuple(capabilities)

    def run(self, task, context=None):
        return AgentResult.ok({"task": task})


def registry():
    return AgentRegistry([
        StubAgent("research", ["research", "evidence"]),
        StubAgent("coding", ["coding", "debug"]),
    ])


def test_research_strategy_wins_for_research_task():
    selection = StrategySelector(registry()).select(Task("t1", "research evidence for this question"))
    assert selection.selected.name == "research-first"
    assert selection.selected.agents == ("research",)


def test_learned_preference_influences_strategy():
    selection = StrategySelector(registry()).select(
        Task("t1", "analyze this complex problem"),
        guidance={"preferred_tools": ["coding"], "avoid_tools": ["research"]},
    )
    assert selection.selected.name == "mixed"


def test_alternatives_are_bounded_and_deterministic():
    selector = StrategySelector(registry())
    first = selector.select(Task("t1", "research and code this"), max_alternatives=1)
    second = selector.select(Task("t1", "research and code this"), max_alternatives=1)
    assert first == second
    assert len(first.alternatives) == 1


def test_dynamic_router_returns_agent_and_selection():
    agent, selection = DynamicAgentRouter(registry()).route(Task("t1", "debug the software"))
    assert agent.name == "coding"
    assert selection.selected.name == "build-first"


def test_unavailable_strategy_fails_closed():
    selector = StrategySelector(
        AgentRegistry([StubAgent("research", ["research"])]),
        strategies=[AgentStrategy("coding-only", ("code",), ("coding",), ("coding",))],
    )
    try:
        selector.select(Task("t1", "code this"))
    except ValueError as exc:
        assert "viable strategy" in str(exc)
    else:
        raise AssertionError("expected strategy selection to fail closed")
