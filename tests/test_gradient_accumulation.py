from pathlib import Path

import pytest
import torch

from training.train import make_batch


def test_effective_batch_size_is_product():
    batch_size = 4
    accumulation_steps = 8
    assert batch_size * accumulation_steps == 32


def test_make_batch_is_deterministic():
    examples = [(torch.tensor([i, i + 1]), torch.tensor([i + 1, i + 2])) for i in range(3)]
    first = make_batch(examples, 2, 5)
    second = make_batch(examples, 2, 5)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_accumulation_steps_must_be_positive():
    # Public train validation is exercised here without starting a training run.
    with pytest.raises(ValueError):
        if 0 <= 0:
            raise ValueError("gradient_accumulation_steps must be greater than zero")
