"""Guarded code execution tool interface for Daweling."""

from typing import Any, Callable

from core.policy import ToolRisk

from .base import BaseTool, ToolResult
from .http_backends import HttpSandboxBackend

RunnerBackend = Callable[[str], Any]


class CodeRunnerTool(BaseTool):
    """Expose an externally sandboxed code runner through the tool contract."""

    name = "code_runner"
    description = "Run code in an isolated sandbox and return execution results."
    risk_level = ToolRisk.APPROVAL_REQUIRED

    def __init__(self, backend: RunnerBackend | None = None, max_code_length: int = 20_000) -> None:
        if max_code_length <= 0:
            raise ValueError("max_code_length must be greater than zero")
        self.backend = backend
        self.max_code_length = max_code_length

    @classmethod
    def from_http(cls, url: str, api_key: str | None = None, timeout: float = 15.0, max_code_length: int = 20_000) -> "CodeRunnerTool":
        return cls(HttpSandboxBackend(url, timeout=timeout, api_key=api_key), max_code_length=max_code_length)

    def run(self, input_data: dict[str, Any]) -> ToolResult:
        code = str(input_data.get("code", ""))
        if not code.strip():
            return ToolResult.fail("Code cannot be empty")
        if len(code) > self.max_code_length:
            return ToolResult.fail("Code exceeds the configured size limit")
        if self.backend is None:
            return ToolResult.fail(
                "No sandboxed code-runner backend is configured",
                capability="code_execution",
            )
        try:
            result = self.backend(code)
        except Exception as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok(result)
