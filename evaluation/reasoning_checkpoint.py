"""Evaluate a reasoning checkpoint on auditable final-answer benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch

from model import DawelingTokenizer, DawelingTransformer, ModelConfig
from .reasoning import ReasoningCase, ReasoningReport, run_reasoning_evaluation


def load_reasoning_checkpoint(path: Path, device: str = "cpu") -> DawelingTransformer:
    """Load a reasoning checkpoint and validate its model configuration."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("reasoning checkpoint must contain a state_dict")
    tokenizer = DawelingTokenizer()
    config_data = checkpoint.get("config")
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    if config_data is not None and config_data != config.__dict__:
        raise ValueError("reasoning checkpoint config does not match the tokenizer/model configuration")
    model = DawelingTransformer(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate_reasoning_checkpoint(
    checkpoint_path: Path,
    cases: Iterable[ReasoningCase],
    predict: Callable[[DawelingTransformer, DawelingTokenizer, str], str],
    *,
    device: str = "cpu",
    name: str = "reasoning-checkpoint",
) -> ReasoningReport:
    """Run the reasoning benchmark against a loaded checkpoint.

    The prediction callback owns generation policy so evaluation remains model-agnostic.
    """
    model = load_reasoning_checkpoint(checkpoint_path, device)
    tokenizer = DawelingTokenizer()
    return run_reasoning_evaluation(
        name,
        cases,
        lambda prompt: predict(model, tokenizer, prompt),
    )
