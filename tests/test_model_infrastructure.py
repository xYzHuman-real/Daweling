import os

from models import ModelConfig, ModelConfigurationError, StaticModelProvider, create_model_provider


def test_model_config_defaults_to_static_without_secrets(monkeypatch):
    monkeypatch.delenv("DAWELING_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("DAWELING_MODEL", raising=False)
    monkeypatch.delenv("DAWELING_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("DAWELING_MODEL_TIMEOUT", raising=False)

    config = ModelConfig.from_env()

    assert config.provider == "static"
    assert config.model == "static"
    assert config.base_url is None
    assert config.timeout_seconds == 30.0


def test_model_config_rejects_non_positive_timeout(monkeypatch):
    monkeypatch.setenv("DAWELING_MODEL_TIMEOUT", "0")

    try:
        ModelConfig.from_env()
    except ValueError as exc:
        assert "greater than zero" in str(exc)
    else:
        raise AssertionError("Expected invalid timeout to fail")


def test_factory_creates_static_provider():
    provider = create_model_provider(ModelConfig(provider="static", model="test"))

    assert isinstance(provider, StaticModelProvider)
    assert provider.model == "test"


def test_factory_rejects_unknown_provider():
    try:
        create_model_provider(ModelConfig(provider="unknown"))
    except ModelConfigurationError as exc:
        assert "Unsupported model provider" in str(exc)
    else:
        raise AssertionError("Expected unsupported provider to fail")


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("DAWELING_MODEL_API_KEY", raising=False)
    from models import OpenAIResponsesProvider

    try:
        OpenAIResponsesProvider(ModelConfig(provider="openai", model="test"))
    except ModelConfigurationError as exc:
        assert "API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing API key to fail")
