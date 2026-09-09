from core.models import Goal, Plan, Task
from planner import ActionBuilder
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class SearchTool(BaseTool):
    name = "search"
    description = "Search the web for information and research a topic."

    def run(self, input_data):
        return ToolResult.ok(input_data)


class CodeTool(BaseTool):
    name = "code"
    description = "Write and inspect software code."

    def run(self, input_data):
        return ToolResult.ok(input_data)


def test_action_builder_selects_tool_by_capability():
    registry = ToolRegistry([SearchTool(), CodeTool()])
    builder = ActionBuilder(registry)
    plan = Plan(
        Goal("Create a website"),
        [Task("research", "Search the web for website requirements")],
    )

    actions = builder.build(plan)

    assert actions[0].tool == "search"
    assert actions[0].input["task"] == "Search the web for website requirements"


def test_action_builder_supports_explicit_tool_override():
    registry = ToolRegistry([SearchTool(), CodeTool()])
    builder = ActionBuilder(registry, explicit_tools={"research": "code"})
    plan = Plan(Goal("research"), [Task("research", "Research something")])

    assert builder.build(plan)[0].tool == "code"


def test_action_builder_rejects_ambiguous_multi_tool_task():
    registry = ToolRegistry([SearchTool(), CodeTool()])
    builder = ActionBuilder(registry)
    plan = Plan(Goal("do something"), [Task("task-1", "Do something")])

    try:
        builder.build(plan)
    except ValueError as exc:
        assert "No suitable tool" in str(exc)
    else:
        raise AssertionError("Expected ambiguous task to fail")
