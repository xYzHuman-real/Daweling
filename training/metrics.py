"""Training metrics and running statistics."""

from __future__ import annotations

import math


def perplexity(loss: float) -> float:
    """Convert average token cross-entropy into perplexity."""
    if loss < 0:
        raise ValueError("loss must be non-negative")
    return math.exp(min(loss, 80.0))


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
