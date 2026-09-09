"""Training schedules used by Daweling's optimizer."""

from __future__ import annotations

import math


def cosine_learning_rate(base_learning_rate: float, step: int, total_steps: int, *, warmup_steps: int = 0, min_learning_rate: float = 0.0) -> float:
    """Return a warmup + cosine-decay learning rate for a training step."""
    if base_learning_rate <= 0 or min_learning_rate < 0 or min_learning_rate > base_learning_rate:
        raise ValueError("learning rates must satisfy 0 <= min_learning_rate <= base_learning_rate")
    if total_steps <= 0 or step < 0 or warmup_steps < 0:
        raise ValueError("steps must be non-negative and total_steps must be positive")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")
    if step < warmup_steps:
        return base_learning_rate * (step + 1) / warmup_steps if warmup_steps else base_learning_rate
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (base_learning_rate - min_learning_rate) * cosine
