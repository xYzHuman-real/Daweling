"""Checkpoint loading and inference helpers for Daweling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .generate import GenerationConfig, generate
from .tokenizer import DawelingTokenizer
from .transformer import DawelingTransformer


def load_checkpoint(model: torch.nn.Module, path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load a Daweling checkpoint using tensor-only deserialization."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model"))
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return checkpoint


def build_model_from_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[DawelingTransformer, DawelingTokenizer, dict[str, Any]]:
    """Construct the correct Daweling model from checkpoint configuration."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint does not contain model configuration")
    config = ModelConfig(**raw_config)
    model = DawelingTransformer(config)
    load_checkpoint(model, path, device=device)
    return model, DawelingTokenizer(), checkpoint


def generate_from_checkpoint(
    checkpoint_path: str | Path,
    prompt: str,
    *,
    config: GenerationConfig | None = None,
    device: str = "cpu",
) -> str:
    """Load a Daweling checkpoint and generate a response to a prompt."""
    model, tokenizer, _ = build_model_from_checkpoint(checkpoint_path, device=device)
    return generate(model, tokenizer, prompt, config)
