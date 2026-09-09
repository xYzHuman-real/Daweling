"""HTTP model provider compatible with OpenAI's Responses API."""

import json
from typing import Any, List
from urllib import error, request

from .base import ModelMessage, ModelProvider, ModelResponse
from .config import ModelConfig
from .errors import ModelConfigurationError, ModelProviderError, ModelResponseError


class OpenAIResponsesProvider(ModelProvider):
    """Minimal dependency-free adapter for the OpenAI Responses API.

    The API key is read from the DAWELING_MODEL_API_KEY environment variable
    and is never stored in repository configuration.
    """

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        import os

        self.config = config
        self.api_key = api_key or os.getenv("DAWELING_MODEL_API_KEY")
        if not self.api_key:
            raise ModelConfigurationError("DAWELING_MODEL_API_KEY is required")
        if not config.model:
            raise ModelConfigurationError("DAWELING_MODEL must not be empty")
        self.endpoint = (config.base_url or "https://api.openai.com/v1/responses").rstrip("/")
        if not self.endpoint.endswith("/responses"):
            self.endpoint += "/responses"

    def generate(self, messages: List[ModelMessage], **kwargs: Any) -> ModelResponse:
        payload = {
            "model": self.config.model,
            "input": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        }
        payload.update(kwargs)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ModelProviderError(f"Model request failed with HTTP {exc.code}: {detail}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise ModelProviderError(f"Model request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelResponseError("Model provider returned invalid JSON") from exc

        content = self._extract_text(data)
        return ModelResponse(content=content, model=str(data.get("model", self.config.model)), metadata=data)

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        text = data.get("output_text")
        if isinstance(text, str) and text:
            return text

        output = data.get("output", [])
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if isinstance(content, dict) and isinstance(content.get("text"), str):
                        chunks.append(content["text"])
            if chunks:
                return "".join(chunks)

        raise ModelResponseError("Model response contained no text output")
