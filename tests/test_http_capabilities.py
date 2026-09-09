import json

from tools import CodeRunnerTool, WebSearchTool
from tools.config import CapabilityConfig
from tools.http_backends import HttpSearchBackend, HttpSandboxBackend


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.payload


def test_http_search_backend_normalizes_results(monkeypatch):
    def fake_urlopen(request, timeout):
        assert request.full_url == "https://search.example/search"
        assert timeout == 3
        return FakeResponse({"results": [{"title": "Daweling"}, "fallback"]})

    monkeypatch.setattr("tools.http_backends.urlopen", fake_urlopen)
    results = HttpSearchBackend("https://search.example/search", timeout=3)("Daweling", 5)
    assert results == [{"title": "Daweling"}, {"value": "fallback"}]


def test_http_sandbox_backend_returns_structured_result(monkeypatch):
    def fake_urlopen(request, timeout):
        assert request.full_url == "https://sandbox.example/run"
        return FakeResponse({"status": "ok", "stdout": "2"})

    monkeypatch.setattr("tools.http_backends.urlopen", fake_urlopen)
    result = HttpSandboxBackend("https://sandbox.example/run")("print(1 + 1)")
    assert result["status"] == "ok"


def test_capability_tools_can_use_http_adapters(monkeypatch):
    monkeypatch.setattr(
        "tools.http_backends.urlopen",
        lambda request, timeout: FakeResponse({"results": [{"title": "ok"}]}),
    )
    search = WebSearchTool.from_http("https://search.example/search")
    result = search.run({"query": "Daweling"})
    assert result.success
    assert result.output == [{"title": "ok"}]


def test_code_runner_http_adapter_is_still_approval_required():
    runner = CodeRunnerTool.from_http("https://sandbox.example/run")
    assert runner.risk_level.value == "approval_required"


def test_capability_config_defaults_to_unconfigured():
    config = CapabilityConfig()
    assert config.web_search_url is None
    assert config.code_runner_url is None
