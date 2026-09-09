from core.models import Action, Goal, Observation, Plan, Task
from models import ModelResponse, ModelProvider
from planner.model_action_builder import ModelActionBuilder
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class ScriptedProvider(ModelProvider):
    def __init__(self, response):
        self.response = response
        self.messages = []

    def generate(self, messages, **kwargs):
        self.messages.append(messages)
        return ModelResponse(content=self.response, model="test")


class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the supplied message."

    def run(self, input_data):
        return ToolResult.ok(input_data.get("message", ""))


def make_plan():
    return Plan(Goal("recover"), [Task("task-1", "Recover the task")])


def test_model_recovery_diagnoses_failure_and_changes_input():
    provider = ScriptedProvider(
        '{"diagnosis":"The first input is invalid; use a corrected message.",'
        '"action":{"task_id":"task-1","tool":"echo","input":{"message":"fixed"}}}'
    )
    builder = ModelActionBuilder(provider, ToolRegistry([EchoTool()]))

    decision = builder.build_recovery_action_with_diagnosis(
        make_plan(), Action("task-1", "echo", {"message": "broken"}),
        Observation("task-1", False, error="invalid input"), 1,
    )

    assert decision.action is not None
    assert decision.diagnosis.startswith("The first input")
    assert decision.action.input == {"message": "fixed"}


def test_model_recovery_can_decline_when_no_safe_fix_exists():
    provider = ScriptedProvider('{"diagnosis":"No safe recovery is available.","action":null}')
    builder = ModelActionBuilder(provider, ToolRegistry([EchoTool()]))
    decision = builder.build_recovery_action_with_diagnosis(
        make_plan(), Action("task-1", "echo", {"message": "broken"}),
        Observation("task-1", False, error="unknown"), 1,
    )
    assert decision.action is None
    assert decision.diagnosis == "No safe recovery is available."
