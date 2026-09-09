"""Environment configuration for external Daweling capability backends."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityConfig:
    """Explicitly configured endpoints for optional external capabilities."""

    web_search_url: str | None = None
    web_search_api_key: str | None = None
    web_search_timeout: float = 10.0
    code_runner_url: str | None = None
    code_runner_api_key: str | None = None
    code_runner_timeout: float = 15.0

    @classmethod
    def from_env(cls) -> "CapabilityConfig":
        return cls(
            web_search_url=os.getenv("DAWELING_WEB_SEARCH_URL"),
            web_search_api_key=os.getenv("DAWELING_WEB_SEARCH_API_KEY"),
            web_search_timeout=float(os.getenv("DAWELING_WEB_SEARCH_TIMEOUT", "10")),
            code_runner_url=os.getenv("DAWELING_CODE_RUNNER_URL"),
            code_runner_api_key=os.getenv("DAWELING_CODE_RUNNER_API_KEY"),
            code_runner_timeout=float(os.getenv("DAWELING_CODE_RUNNER_TIMEOUT", "15")),
        )
