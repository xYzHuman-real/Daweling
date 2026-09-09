from core.models import Action, Goal, Plan, Task
from core.runtime import Runtime
from orchestrator.loop import DecisionDrivenLoop
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class FlakyTool(BaseTool):
    name = "flaky"
    description = "Fails once, then succeeds"
    calls = 0

    def run(self, input_data):
        type(self).calls += 1
        if type(self).calls == 1:
            return ToolResult.fail("temporary failure")
        return ToolResult.ok("recovered")


class OneTaskPlanner:
    def create_plan(self, goal):
        return Plan(goal, [Task("task-1", "perform task")])


def test_decision_driven_loop_recovers_and_completes():
    FlakyTool.calls = 0
    loop = DecisionDrivenLoop(Runtime(ToolRegistry([FlakyTool()])), planner=OneTaskPlanner(), max_recovery_attempts=1)
    actions = [Action("task-1", "flaky", {})]

    result = loop.run(
        Goal("test"),
        lambda plan: actions,
        recover=lambda plan, action, observation: action,
    )

    assert result.success
    assert result.recovery_attempts == 1
    assert [decision.next_step.value for decision in result.decisions] == ["recover", "complete"]
    assert result.observations[-1].success


def test_review_rejection_is_bounded_and_fails_without_replan_handler():
    loop = DecisionDrivenLoop(Runtime(ToolRegistry([FlakyTool()])), planner=OneTaskPlanner())
    result = loop.run(
        Goal("test"),
        lambda plan: [Action("task-1", "flaky", {})],
        review=lambda plan, observations, verifications: False,
        review_required=True,
    )

    assert not result.success
    assert result.decisions[-1].next_step.value == "fail"
