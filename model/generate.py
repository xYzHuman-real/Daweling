"""Text generation utilities for the Daweling decoder model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tokenizer import DawelingTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = 40
    do_sample: bool = True


def _sample_next(logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    if config.temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    logits = logits / config.temperature
    if config.top_k is not None:
        if config.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        k = min(config.top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < values[..., -1, None], float("-inf"))
    if not config.do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def generate(
    model: torch.nn.Module,
    tokenizer: DawelingTokenizer,
    prompt: str,
    config: GenerationConfig | None = None,
) -> str:
    """Generate text from a causal Daweling model using next-token prediction."""
    config = config or GenerationConfig()
    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    model_config = getattr(model, "config", None)
    context_limit = getattr(model_config, "max_sequence_length", None)
    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            context = tokens[:, -context_limit:] if context_limit else tokens
            output = model(context)
            logits = output[0] if isinstance(output, tuple) else getattr(output, "logits", output)
            next_token = _sample_next(logits[:, -1, :], config)
            tokens = torch.cat((tokens, next_token), dim=1)
            if next_token.item() == tokenizer.eos_id:
                break

    return tokenizer.decode(tokens[0].tolist())
