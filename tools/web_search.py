"""Controlled web-search tool interface for Daweling."""

from typing import Any, Callable

from core.policy import ToolRisk

from .base import BaseTool, ToolResult
from .http_backends import HttpSearchBackend

SearchBackend = Callable[[str, int], Any]


class WebSearchTool(BaseTool):
    """Expose a configured web-search backend through Daweling's tool contract."""

    name = "web_search"
    description = "Search the web for current information and return structured results."
    risk_level = ToolRisk.SAFE

    def __init__(self, backend: SearchBackend | None = None, max_results: int = 5) -> None:
        if max_results <= 0:
            raise ValueError("max_results must be greater than zero")
        self.backend = backend
        self.max_results = max_results

    @classmethod
    def from_http(cls, url: str, api_key: str | None = None, timeout: float = 10.0, max_results: int = 5) -> "WebSearchTool":
        return cls(HttpSearchBackend(url, timeout=timeout, api_key=api_key), max_results=max_results)

    def run(self, input_data: dict[str, Any]) -> ToolResult:
        query = str(input_data.get("query", "")).strip()
        if not query:
            return ToolResult.fail("Search query cannot be empty")
        if self.backend is None:
            return ToolResult.fail(
                "No web-search backend is configured",
                query=query,
                capability="web_search",
            )
        try:
            results = self.backend(query, self.max_results)
        except Exception as exc:
            return ToolResult.fail(str(exc), query=query)
        return ToolResult.ok(results, query=query, result_count=self._count(results))

    @staticmethod
    def _count(results: Any) -> int:
        try:
            return len(results)
        except TypeError:
            return 0
