"""Environment-driven model configuration for Daweling."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelConfig:
    """Configuration that contains no secret values by default."""

    provider: str = "static"
    model: str = "static"
    base_url: str | None = None
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "ModelConfig":
        timeout = float(os.getenv("DAWELING_MODEL_TIMEOUT", "30"))
        if timeout <= 0:
            raise ValueError("DAWELING_MODEL_TIMEOUT must be greater than zero")
        return cls(
            provider=os.getenv("DAWELING_MODEL_PROVIDER", "static").strip().lower(),
            model=os.getenv("DAWELING_MODEL", "static").strip(),
            base_url=os.getenv("DAWELING_MODEL_BASE_URL") or None,
            timeout_seconds=timeout,
        )
