from agents.base import AgentResult, BaseAgent
from agents.collaboration import AgentCollaborator
from agents.debate import AgentDebate
from agents.pipeline import AgentPipeline
from agents.registry import AgentRegistry
from agents.router import AgentRouter
from core.models import Goal, Plan, Task
from core.runtime import Runtime
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class ResearchAgent(BaseAgent):
    name = "research"
    capabilities = ("research",)
    description = "Research"

    def run(self, task, context=None):
        return AgentResult.ok("evidence")


class ReviewAgent(BaseAgent):
    name = "review"
    capabilities = ("review",)
    description = "Review"

    def run(self, task, context=None):
        return AgentResult.ok({"approved": True, "critique": "Verified."})


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo"

    def run(self, input_data):
        return ToolResult.ok(input_data["message"])


class OneTaskPlanner:
    def create_plan(self, goal):
        return Plan(goal, [Task("task-1", "research the answer")])


def test_pipeline_connects_collaboration_execution_verification_and_review():
    registry = AgentRegistry([ResearchAgent(), ReviewAgent()])
    collaborator = AgentCollaborator(AgentRouter(registry))
    debate = AgentDebate([registry.require("review")])
    pipeline = AgentPipeline(collaborator, Runtime(ToolRegistry([EchoTool()])), planner=OneTaskPlanner(), debate=debate)

    result = pipeline.run(
        Goal("answer"),
        lambda plan, context: [__import__("core.models", fromlist=["Action"]).Action("task-1", "echo", {"message": context["task-1"]["output"]})],
    )

    assert result.collaboration.context["task-1"]["output"] == "evidence"
    assert result.observations[0].output == "evidence"
    assert result.verifications[0].valid
    assert result.review is not None and result.review.accepted
    assert result.success
