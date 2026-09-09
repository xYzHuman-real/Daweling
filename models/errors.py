"""Errors raised by model provider infrastructure."""


class ModelProviderError(RuntimeError):
    """Base error for failures originating in a model provider."""


class ModelConfigurationError(ModelProviderError):
    """Raised when provider configuration is invalid."""


class ModelResponseError(ModelProviderError):
    """Raised when a provider returns an unusable response."""
