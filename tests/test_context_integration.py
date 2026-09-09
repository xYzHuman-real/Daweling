from core.models import Goal
from daweling import Daweling
from memory import MemoryStore
from models import ModelMessage, ModelProvider, ModelResponse
from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class CapturingProvider(ModelProvider):
    def __init__(self):
        self.calls = []
        self.responses = iter([
            '{"tasks":[{"id":"task-1","description":"Use the stored architecture"}]}',
            '{"actions":[{"task_id":"task-1","tool":"echo","input":{"message":"ok"}}]}',
        ])

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return ModelResponse(content=next(self.responses), model="test")


class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the supplied message."

    def run(self, input_data):
        return ToolResult.ok(input_data.get("message", ""))


def test_daweling_injects_relevant_memory_into_planning(tmp_path):
    memory = MemoryStore(tmp_path / "memory.json")
    memory.remember("architecture", "Daweling uses an execution-first architecture", category="project")
    provider = CapturingProvider()

    result = Daweling(provider, ToolRegistry([EchoTool()]), memory=memory).run(
        Goal("Build Daweling architecture")
    )

    assert result.success is True
    planning_messages = provider.calls[0]
    assert any("execution-first architecture" in message.content for message in planning_messages)
