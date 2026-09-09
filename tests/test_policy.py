from core.models import Action, Goal, Plan, Task
from core.policy import ApprovalPolicy, ToolRisk
from core.runtime import Runtime
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class SensitiveTool(BaseTool):
    name = "sensitive"
    description = "Represents a consequential action."
    risk_level = ToolRisk.APPROVAL_REQUIRED

    def __init__(self):
        self.calls = 0

    def run(self, input_data):
        self.calls += 1
        return ToolResult.ok(input_data.get("value"))


def make_plan():
    return Plan(Goal("perform sensitive action"), [Task("task-1", "Perform sensitive action")])


def test_approval_required_tool_is_blocked_without_approval():
    tool = SensitiveTool()
    runtime = Runtime(ToolRegistry([tool]))

    observations = runtime.execute(make_plan(), [Action("task-1", "sensitive", {"value": "x"})])

    assert observations[0].success is False
    assert "Approval required" in observations[0].error
    assert tool.calls == 0


def test_approval_required_tool_runs_when_policy_approves():
    tool = SensitiveTool()
    policy = ApprovalPolicy(approval_callback=lambda name, reason: name == "sensitive")
    runtime = Runtime(ToolRegistry([tool]), approval_policy=policy)

    observations = runtime.execute(make_plan(), [Action("task-1", "sensitive", {"value": "approved"})])

    assert observations[0].success is True
    assert observations[0].output == "approved"
    assert tool.calls == 1
