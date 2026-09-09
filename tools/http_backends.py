"""HTTP adapters for externally hosted Daweling capabilities.

These adapters keep network access outside the tool classes themselves and
require explicitly configured endpoints. No credentials are hard-coded.
"""

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HttpBackendError(RuntimeError):
    """Raised when a configured capability endpoint cannot be reached safely."""


def _post_json(url: str, payload: dict[str, Any], timeout: float, api_key: str | None = None) -> Any:
    if not url.strip():
        raise HttpBackendError("HTTP backend URL is not configured")
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(1_000_001)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise HttpBackendError(f"Capability endpoint request failed: {exc}") from exc
    if len(raw) > 1_000_000:
        raise HttpBackendError("Capability endpoint response is too large")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HttpBackendError("Capability endpoint returned invalid JSON") from exc


class HttpSearchBackend:
    """Call a configured search service and normalize common result formats."""

    def __init__(self, url: str, timeout: float = 10.0, api_key: str | None = None) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.url = url
        self.timeout = timeout
        self.api_key = api_key

    def __call__(self, query: str, limit: int) -> list[dict[str, Any]]:
        data = _post_json(
            self.url,
            {"query": query, "limit": limit},
            self.timeout,
            self.api_key,
        )
        results = data.get("results") if isinstance(data, dict) else data
        if not isinstance(results, list):
            raise HttpBackendError("Search endpoint response must contain a results list")
        normalized: list[dict[str, Any]] = []
        for item in results[:limit]:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"value": item})
        return normalized


class HttpSandboxBackend:
    """Call a configured isolated code-execution service.

    Daweling never executes the submitted code locally. The remote service is
    responsible for sandboxing, resource limits, and language policy.
    """

    def __init__(self, url: str, timeout: float = 15.0, api_key: str | None = None) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.url = url
        self.timeout = timeout
        self.api_key = api_key

    def __call__(self, code: str) -> Any:
        data = _post_json(self.url, {"code": code}, self.timeout, self.api_key)
        if not isinstance(data, dict):
            raise HttpBackendError("Sandbox endpoint response must be a JSON object")
        return data
