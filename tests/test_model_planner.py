from core.models import Goal
from models import ModelMessage, ModelProvider, ModelResponse
from planner.model_planner import ModelPlanner


class FakeProvider(ModelProvider):
    def __init__(self, content):
        self.content = content
        self.messages = None

    def generate(self, messages, **kwargs):
        self.messages = messages
        return ModelResponse(content=self.content, model="fake")


def test_model_planner_builds_ordered_tasks():
    provider = FakeProvider(
        '{"tasks": [{"id": "research", "description": "Research the topic"}, '
        '{"id": "write", "description": "Write the result"}]}'
    )
    planner = ModelPlanner(provider)

    plan = planner.create_plan(Goal("Create a research brief"))

    assert [task.id for task in plan.tasks] == ["research", "write"]
    assert plan.tasks[1].description == "Write the result"
    assert provider.messages[0].role == "system"
    assert provider.messages[1].role == "user"


def test_model_planner_rejects_invalid_json():
    planner = ModelPlanner(FakeProvider("not json"))

    try:
        planner.create_plan(Goal("test"))
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError("Expected invalid model output to fail")


def test_model_planner_rejects_duplicate_ids():
    provider = FakeProvider(
        '{"tasks": [{"id": "same", "description": "first"}, '
        '{"id": "same", "description": "second"}]}'
    )

    try:
        ModelPlanner(provider).create_plan(Goal("test"))
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate ids to fail")
