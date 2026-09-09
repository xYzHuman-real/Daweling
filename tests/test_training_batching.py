import torch

from model import DawelingTokenizer
from training.train import make_batch, make_examples


def test_make_batch_returns_requested_batch_size():
    tokenizer = DawelingTokenizer()
    examples = list(make_examples("hello world " * 80, tokenizer, 16))
    inputs, targets = make_batch(examples, batch_size=4, step=0)

    assert inputs.shape == (4, 16)
    assert targets.shape == (4, 16)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_make_batch_is_deterministic_and_cycles_examples():
    tokenizer = DawelingTokenizer()
    examples = list(make_examples("abcdef " * 80, tokenizer, 8))

    first = make_batch(examples, batch_size=3, step=2)
    second = make_batch(examples, batch_size=3, step=2)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
