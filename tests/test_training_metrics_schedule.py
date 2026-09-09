import pytest

from training.metrics import perplexity
from training.schedule import cosine_learning_rate


def test_perplexity_matches_exponential_loss():
    assert perplexity(0.0) == 1.0


def test_cosine_schedule_warms_up_and_decays():
    values = [cosine_learning_rate(1e-3, step, 10, warmup_steps=2, min_learning_rate=1e-4) for step in range(10)]
    assert values[0] < values[1]
    assert values[2] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-4)
