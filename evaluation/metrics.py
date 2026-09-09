"""Small dependency-free metrics used by Daweling evaluations."""

from __future__ import annotations

import math
from collections.abc import Iterable


def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())


def mean_score(scores: Iterable[float]) -> float:
    values = list(scores)
    if not values:
        raise ValueError("scores cannot be empty")
    return sum(values) / len(values)


def perplexity(mean_negative_log_likelihood: float) -> float:
    """Convert mean token negative log likelihood to perplexity."""
    if not math.isfinite(mean_negative_log_likelihood):
        raise ValueError("negative log likelihood must be finite")
    return math.exp(mean_negative_log_likelihood)
