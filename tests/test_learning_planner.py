from core.models import Goal
from memory import MemoryStore
from memory.context import ContextEngine
from models import ModelMessage, ModelProvider, ModelResponse
from planner.model_planner import ModelPlanner


class CapturingPlannerProvider(ModelProvider):
    def __init__(self):
        self.messages: list[ModelMessage] = []

    def generate(self, messages, **kwargs):
        self.messages = list(messages)
        return ModelResponse(
            content='{"tasks":[{"id":"task-1","description":"Use the proven approach"}]}',
            model="test",
        )


def test_model_planner_receives_past_experience(tmp_path):
    memory = MemoryStore(tmp_path / "memory.json")
    memory.remember(
        "experience:success",
        {
            "goal": "Build Daweling architecture",
            "success": True,
            "successful_tasks": 2,
            "failed_tasks": 0,
            "tools_used": ["architecture_tool"],
            "verification_reason": "verified",
        },
        category="experience",
    )

    provider = CapturingPlannerProvider()
    context = ContextEngine(memory).build(Goal("Build Daweling architecture"))
    ModelPlanner(provider).create_plan(Goal("Build Daweling architecture"), context=context)

    combined = "\n".join(message.content for message in provider.messages)
    assert "past workflow experience" in combined
    assert "architecture_tool" in combined
    assert "success" in combined
