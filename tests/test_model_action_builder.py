from core.models import Goal, Plan, Task
from models import StaticModelProvider
from planner.model_action_builder import ModelActionBuilder
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the supplied message."

    def run(self, input_data):
        return ToolResult.ok(input_data.get("message", ""))


def make_plan():
    return Plan(Goal("say hello"), [Task("task-1", "Say hello")])


def test_model_action_builder_creates_valid_actions():
    provider = StaticModelProvider(
        '{"actions":[{"task_id":"task-1","tool":"echo","input":{"message":"hello"}}]}'
    )
    builder = ModelActionBuilder(provider, ToolRegistry([EchoTool()]))

    actions = builder.build_actions(make_plan())

    assert len(actions) == 1
    assert actions[0].task_id == "task-1"
    assert actions[0].tool == "echo"
    assert actions[0].input["message"] == "hello"


def test_model_action_builder_rejects_unavailable_tool():
    provider = StaticModelProvider(
        '{"actions":[{"task_id":"task-1","tool":"missing","input":{}}]}'
    )

    try:
        ModelActionBuilder(provider, ToolRegistry([EchoTool()])).build_actions(make_plan())
    except ValueError as exc:
        assert "unavailable tool" in str(exc)
    else:
        raise AssertionError("Expected unavailable tool to fail")


def test_model_action_builder_rejects_unknown_task():
    provider = StaticModelProvider(
        '{"actions":[{"task_id":"missing","tool":"echo","input":{}}]}'
    )

    try:
        ModelActionBuilder(provider, ToolRegistry([EchoTool()])).build_actions(make_plan())
    except ValueError as exc:
        assert "unknown task" in str(exc)
    else:
        raise AssertionError("Expected unknown task to fail")
