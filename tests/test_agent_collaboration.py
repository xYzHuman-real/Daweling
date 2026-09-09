from agents import AgentRegistry, AgentRouter
from agents.base import AgentResult, BaseAgent
from agents.collaboration import AgentCollaborator
from core.models import Goal, Plan, Task


class ResearchAgent(BaseAgent):
    name = "research"
    description = "Research specialist"
    capabilities = ("research",)

    def __init__(self):
        self.contexts = []

    def run(self, task, context=None):
        self.contexts.append(context or {})
        return AgentResult.ok("research evidence")


class CodingAgent(BaseAgent):
    name = "coding"
    description = "Coding specialist"
    capabilities = ("coding",)

    def __init__(self):
        self.contexts = []

    def run(self, task, context=None):
        self.contexts.append(context or {})
        return AgentResult.ok("code based on evidence")


def test_agents_collaborate_through_shared_context():
    research = ResearchAgent()
    coding = CodingAgent()
    router = AgentRouter(AgentRegistry([research, coding]))
    collaborator = AgentCollaborator(router)
    plan = Plan(Goal("research and code"), [
        Task("task-1", "research the problem"),
        Task("task-2", "implement the solution using the research"),
    ])

    result = collaborator.collaborate(plan)

    assert [step.agent for step in result.steps] == ["research", "coding"]
    assert result.success
    assert coding.contexts[0]["task-1"]["output"] == "research evidence"
    assert result.context["task-2"]["output"] == "code based on evidence"


def test_collaboration_bounds_shared_context():
    research = ResearchAgent()
    router = AgentRouter(AgentRegistry([research]))
    collaborator = AgentCollaborator(router, max_context_items=1)
    plan = Plan(Goal("research"), [Task("task-1", "research this")])
    result = collaborator.collaborate(plan, {"old": "context", "new": "context"})

    assert result.steps[0].result.success
    assert list(research.contexts[0]) == ["new"]
