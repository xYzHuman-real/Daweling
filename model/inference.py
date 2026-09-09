"""Checkpoint loading and inference helpers for Daweling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .generate import GenerationConfig, generate
from .tokenizer import DawelingTokenizer


def load_checkpoint(model: torch.nn.Module, path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load a Daweling checkpoint without executing arbitrary serialized code."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return checkpoint if isinstance(checkpoint, dict) else {"model": state_dict}


def generate_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    prompt: str,
    *,
    tokenizer: DawelingTokenizer | None = None,
    config: GenerationConfig | None = None,
    device: str = "cpu",
) -> str:
    tokenizer = tokenizer or DawelingTokenizer()
    load_checkpoint(model, checkpoint_path, device=device)
    return generate(model, tokenizer, prompt, config)
