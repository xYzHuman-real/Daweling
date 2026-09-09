from core.models import Action, Goal
from daweling import Daweling
from models import ModelMessage, ModelProvider, ModelResponse
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class ScriptedProvider(ModelProvider):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def generate(self, messages, **kwargs):
        self.messages.append(messages)
        return ModelResponse(content=next(self.responses), model="test")


class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the supplied message."

    def run(self, input_data):
        return ToolResult.ok(input_data.get("message", ""))


class RecoverableTool(BaseTool):
    name = "recoverable"
    description = "Fails on the first call and succeeds on the next call."

    def __init__(self):
        self.calls = 0

    def run(self, input_data):
        self.calls += 1
        if self.calls == 1:
            return ToolResult.fail("temporary problem")
        return ToolResult.ok(input_data.get("message", "recovered"))


def test_daweling_runs_model_to_verified_result():
    provider = ScriptedProvider(
        [
            '{"tasks":[{"id":"task-1","description":"Say hello"}]}',
            '{"actions":[{"task_id":"task-1","tool":"echo","input":{"message":"hello from Daweling"}}]}',
        ]
    )
    daweling = Daweling(provider, ToolRegistry([EchoTool()]))

    result = daweling.run(Goal("Say hello"))

    assert result.success is True
    assert result.observations[0].output == "hello from Daweling"
    assert result.verifications[0].valid is True
    assert len(provider.messages) == 2


def test_daweling_keeps_model_selection_separate_from_execution():
    provider = ScriptedProvider(
        [
            '{"tasks":[{"id":"task-1","description":"Do work"}]}',
            '{"actions":[{"task_id":"task-1","tool":"missing","input":{}}]}',
        ]
    )
    daweling = Daweling(provider, ToolRegistry([EchoTool()]))

    try:
        daweling.run(Goal("Do work"))
    except ValueError as exc:
        assert "unavailable tool" in str(exc)
    else:
        raise AssertionError("Expected unavailable tool selection to fail before execution")


def test_daweling_can_recover_from_a_failed_tool_action():
    provider = ScriptedProvider(
        [
            '{"tasks":[{"id":"task-1","description":"Recover"}]}',
            '{"actions":[{"task_id":"task-1","tool":"recoverable","input":{"message":"first"}}]}',
        ]
    )
    tool = RecoverableTool()
    daweling = Daweling(provider, ToolRegistry([tool]))

    result = daweling.run_with_recovery(
        Goal("Recover"),
        lambda action, _observation, _attempt: Action(
            action.task_id,
            action.tool,
            {"message": "recovered"},
        ),
    )

    assert result.success is True
    assert result.observations[0].output == "recovered"
    assert tool.calls == 2
