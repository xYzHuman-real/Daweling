from core.reasoning import ReasoningEngine
from models import ModelProvider, ModelResponse


class FakeProvider(ModelProvider):
    def __init__(self, content):
        self.content = content
        self.messages = None

    def generate(self, messages, **kwargs):
        self.messages = messages
        return ModelResponse(self.content, model="fake")


def test_reasoning_engine_returns_structured_result():
    provider = FakeProvider(
        '{"understanding":"Need a verified answer","strategy":"research first",'
        '"steps":[{"purpose":"identify evidence","conclusion":"collect relevant sources"}],'
        '"uncertainties":["source quality"]}'
    )
    result = ReasoningEngine(provider).reason("Answer the question", {"topic": "AI"})

    assert result.understanding == "Need a verified answer"
    assert result.strategy == "research first"
    assert result.steps[0].purpose == "identify evidence"
    assert result.uncertainties == ("source quality",)
    assert result.as_dict()["steps"][0]["conclusion"] == "collect relevant sources"


def test_reasoning_engine_rejects_invalid_json():
    try:
        ReasoningEngine(FakeProvider("not json")).reason("test")
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError("Expected invalid JSON to fail")


def test_reasoning_engine_enforces_step_bound():
    content = '{"understanding":"u","strategy":"s","steps":[' + ",".join(
        '{"purpose":"p","conclusion":"c"}' for _ in range(3)
    ) + '],"uncertainties":[]}'
    try:
        ReasoningEngine(FakeProvider(content), max_steps=2).reason("test")
    except ValueError as exc:
        assert "bounds" in str(exc)
    else:
        raise AssertionError("Expected reasoning bounds to fail")


def test_reasoning_engine_does_not_request_private_chain_of_thought():
    provider = FakeProvider(
        '{"understanding":"u","strategy":"s","steps":[{"purpose":"p","conclusion":"c"}],"uncertainties":[]}'
    )
    ReasoningEngine(provider).reason("test")
    system_prompt = provider.messages[0].content
    assert "Do not reveal private chain-of-thought" in system_prompt
