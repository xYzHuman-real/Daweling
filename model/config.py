"""Configuration for the first Daweling language-model experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Small decoder-only Transformer configuration for local experiments."""

    vocab_size: int = 512
    max_sequence_length: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.vocab_size, self.max_sequence_length, self.d_model, self.n_heads, self.n_layers) <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
