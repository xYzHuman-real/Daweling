from tools import CodeRunnerTool, WebSearchTool


def test_web_search_requires_backend():
    result = WebSearchTool().run({"query": "Daweling"})
    assert not result.success
    assert "backend" in (result.error or "")


def test_web_search_uses_backend():
    tool = WebSearchTool(lambda query, limit: [{"title": query}], max_results=3)
    result = tool.run({"query": "Daweling"})
    assert result.success
    assert result.output == [{"title": "Daweling"}]
    assert result.metadata["result_count"] == 1


def test_code_runner_requires_backend():
    result = CodeRunnerTool().run({"code": "print(1)"})
    assert not result.success
    assert "sandboxed" in (result.error or "")


def test_code_runner_uses_backend():
    tool = CodeRunnerTool(lambda code: {"ran": code})
    result = tool.run({"code": "print(1)"})
    assert result.success
    assert result.output == {"ran": "print(1)"}


def test_code_runner_rejects_oversized_code():
    tool = CodeRunnerTool(lambda code: code, max_code_length=3)
    result = tool.run({"code": "1234"})
    assert not result.success
    assert "size limit" in (result.error or "")
