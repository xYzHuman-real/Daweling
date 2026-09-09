"""Factory for selecting a configured model provider."""

from .config import ModelConfig
from .errors import ModelConfigurationError
from .mock import StaticModelProvider
from .openai_compatible import OpenAIResponsesProvider


def create_model_provider(config: ModelConfig):
    """Create the provider selected by configuration."""
    if config.provider == "static":
        return StaticModelProvider("{\"tasks\":[{\"id\":\"task-1\",\"description\":\"Complete the requested goal\"}]}", config.model)
    if config.provider in {"openai", "openai-responses"}:
        return OpenAIResponsesProvider(config)
    raise ModelConfigurationError(f"Unsupported model provider: {config.provider}")
