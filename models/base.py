"""Provider-agnostic interfaces for model-backed Daweling components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ModelMessage:
    """A single message in a model conversation."""

    role: str
    content: str


@dataclass(frozen=True)
class ModelResponse:
    """Normalized response returned by a model provider."""

    content: str
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """Contract that any LLM/model backend can implement."""

    @abstractmethod
    def generate(self, messages: List[ModelMessage], **kwargs: Any) -> ModelResponse:
        """Generate a response from a sequence of messages."""
        raise NotImplementedError
