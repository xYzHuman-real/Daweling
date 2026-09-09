from core.models import Action, Goal
from core.runtime import Runtime
from orchestrator import Orchestrator
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the supplied message."

    def run(self, input_data):
        return ToolResult.ok(input_data.get("message", ""))


def test_orchestrator_runs_full_core_loop():
    registry = ToolRegistry([EchoTool()])
    orchestrator = Orchestrator(runtime=Runtime(registry))

    result = orchestrator.run(
        Goal("Build a landing page"),
        lambda plan: [
            Action(
                task_id=plan.tasks[0].id,
                tool="echo",
                input={"message": plan.goal.description},
            )
        ],
    )

    assert result.success is True
    assert result.plan.tasks[0].status.value == "completed"
    assert result.observations[0].output == "Build a landing page"
    assert result.verifications[0].valid is True


def test_orchestrator_rejects_action_for_unknown_task():
    orchestrator = Orchestrator()

    try:
        orchestrator.run(
            Goal("test"),
            lambda _plan: [Action("missing", "echo")],
        )
    except ValueError as exc:
        assert "unknown task" in str(exc).lower()
    else:
        raise AssertionError("Expected unknown task action to fail")


def test_orchestrator_reports_tool_failure_as_unverified():
    registry = ToolRegistry([EchoTool()])
    orchestrator = Orchestrator(runtime=Runtime(registry))

    result = orchestrator.run(
        Goal("test"),
        lambda plan: [Action(plan.tasks[0].id, "missing-tool")],
    )

    assert result.success is False
    assert result.observations[0].success is False
    assert result.verifications[0].valid is False
