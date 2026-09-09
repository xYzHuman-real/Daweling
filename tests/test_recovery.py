from core.models import Action, Goal, Plan, Task
from core.recovery import RecoveryEngine
from core.runtime import Runtime
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class FlakyTool(BaseTool):
    name = "flaky"
    description = "Fails once, then succeeds."

    def __init__(self):
        self.calls = 0

    def run(self, input_data):
        self.calls += 1
        if self.calls == 1:
            return ToolResult.fail("temporary failure")
        return ToolResult.ok("recovered")


class AlwaysFailTool(BaseTool):
    name = "fail"
    description = "Always fails."

    def run(self, input_data):
        return ToolResult.fail("still broken")


def make_plan():
    return Plan(goal=Goal("test"), tasks=[Task("task-1", "test")])


def test_recovery_retries_with_replacement_action():
    tool = FlakyTool()
    runtime = Runtime(ToolRegistry([tool]))
    engine = RecoveryEngine(runtime, max_attempts=2)

    result = engine.execute(
        make_plan(),
        Action("task-1", "flaky", {"attempt": 1}),
        recover=lambda action, observation, attempt: Action(
            action.task_id, action.tool, {"attempt": attempt + 1}
        ),
    )

    assert result.success is True
    assert [observation.success for observation in result.observations] == [False, True]
    assert len(result.attempts) == 1
    assert result.attempts[0].replacement_action.input == {"attempt": 2}


def test_recovery_stops_when_policy_declines():
    runtime = Runtime(ToolRegistry([AlwaysFailTool()]))
    engine = RecoveryEngine(runtime, max_attempts=3)

    result = engine.execute(
        make_plan(),
        Action("task-1", "fail"),
        recover=lambda *_: None,
    )

    assert result.success is False
    assert len(result.observations) == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].replacement_action is None


def test_recovery_is_bounded():
    runtime = Runtime(ToolRegistry([AlwaysFailTool()]))
    engine = RecoveryEngine(runtime, max_attempts=2)

    result = engine.execute(
        make_plan(),
        Action("task-1", "fail"),
        recover=lambda action, _observation, _attempt: action,
    )

    assert len(result.observations) == 3
    assert len(result.attempts) == 2
