"""Deterministic model provider useful for tests and local development."""

from typing import Any, List

from .base import ModelMessage, ModelProvider, ModelResponse


class StaticModelProvider(ModelProvider):
    """Return a predefined response without contacting an external service."""

    def __init__(self, content: str, model: str = "static") -> None:
        self.content = content
        self.model = model
        self.last_messages: List[ModelMessage] = []

    def generate(self, messages: List[ModelMessage], **kwargs: Any) -> ModelResponse:
        self.last_messages = list(messages)
        return ModelResponse(content=self.content, model=self.model)
