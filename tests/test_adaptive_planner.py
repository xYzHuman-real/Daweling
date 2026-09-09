from core.models import Action, Goal, Observation, Plan, Task, VerificationResult
from models import ModelMessage, ModelProvider, ModelResponse
from planner.adaptive_planner import AdaptivePlanner


class ScriptedProvider(ModelProvider):
    def __init__(self, content):
        self.content = content
        self.messages = []

    def generate(self, messages, **kwargs):
        self.messages.append(messages)
        return ModelResponse(content=self.content, model="test")


def test_adaptive_planner_replaces_failed_plan_with_evidence_based_plan():
    provider = ScriptedProvider(
        '{"reason":"The first approach used the wrong input; simplify the task.",'
        '"tasks":[{"id":"task-1","description":"Retry with corrected input"},'
        '{"id":"task-2","description":"Verify the corrected result"}]}'
    )
    planner = AdaptivePlanner(provider)
    plan = Plan(Goal("finish the job"), [Task("task-1", "Do the job")])
    result = planner.replan(
        plan,
        [Observation("task-1", False, error="invalid input")],
        [VerificationResult.failed("Output was invalid")],
        [Action("task-1", "echo", {"message": "bad"})],
    )

    assert result.reason.startswith("The first approach")
    assert [task.id for task in result.plan.tasks] == ["task-1", "task-2"]
    assert "invalid input" in provider.messages[1].content


def test_adaptive_planner_does_not_replan_successful_work():
    provider = ScriptedProvider("{}"); planner = AdaptivePlanner(provider)
    plan = Plan(Goal("done"), [Task("task-1", "Do it")])
    result = planner.replan(
        plan,
        [Observation("task-1", True, output="ok")],
        [VerificationResult.passed()],
        [Action("task-1", "echo", {"message": "ok"})],
    )
    assert result.plan is plan
    assert provider.messages == []
