from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the supplied message."

    def run(self, input_data):
        return ToolResult.ok(input_data.get("message", ""))


def test_registry_registers_and_finds_tools():
    registry = ToolRegistry([EchoTool()])

    assert registry.names() == ["echo"]
    assert registry.get("echo").name == "echo"
    assert registry.require("echo").description == "Returns the supplied message."
    assert len(registry) == 1


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry([EchoTool()])

    try:
        registry.register(EchoTool())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Expected duplicate registration to fail")


def test_tool_result_helpers():
    success = ToolResult.ok("hello", source="test")
    failure = ToolResult.fail("boom", retryable=True)

    assert success.success is True
    assert success.output == "hello"
    assert success.metadata["source"] == "test"
    assert failure.success is False
    assert failure.error == "boom"
    assert failure.metadata["retryable"] is True
