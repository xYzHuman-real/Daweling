from agents import AgentRegistry, AgentResult, BaseAgent
from agents.router import AgentRouter
from core.models import Task


class FakeAgent(BaseAgent):
    def __init__(self, name, capabilities):
        self.name = name
        self.description = name
        self.capabilities = capabilities

    def run(self, task, context=None):
        return AgentResult.ok(task)


def test_agent_registry_registers_and_retrieves_agents():
    agent = FakeAgent("research", ("research", "evidence"))
    registry = AgentRegistry([agent])

    assert registry.require("research") is agent
    assert registry.names() == ["research"]


def test_agent_router_selects_research_agent():
    research = FakeAgent("research", ("research", "evidence"))
    coding = FakeAgent("coding", ("coding", "debugging"))
    router = AgentRouter(AgentRegistry([research, coding]))

    selected = router.route(Task("task-1", "Research evidence for this question"))

    assert selected is research


def test_agent_router_selects_coding_agent():
    research = FakeAgent("research", ("research", "evidence"))
    coding = FakeAgent("coding", ("coding", "debugging"))
    router = AgentRouter(AgentRegistry([research, coding]))

    selected = router.route(Task("task-1", "Debug this Python implementation"))

    assert selected is coding


def test_agent_router_returns_none_when_no_specialization_matches():
    router = AgentRouter(AgentRegistry([FakeAgent("research", ("research",))]))

    assert router.route(Task("task-1", "Make breakfast")) is None
