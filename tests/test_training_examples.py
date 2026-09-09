import torch

from model import DawelingTokenizer
from training.train import make_examples_from_texts


def test_dataset_examples_do_not_cross_boundaries():
    tokenizer = DawelingTokenizer()
    sequence_length = 4
    first = "abcdefghij"
    second = "KLMNOPQRST"

    examples = make_examples_from_texts((first, second), tokenizer, sequence_length)
    second_ids = tokenizer.encode(second)

    # Every produced target sequence must originate inside one source example.
    second_start = tuple(second_ids[1:sequence_length + 1])
    assert all(not torch.equal(target, torch.tensor(second_start, dtype=torch.long)) for _, target in examples if target.numel() == sequence_length)


def test_empty_or_short_dataset_examples_are_skipped():
    tokenizer = DawelingTokenizer()
    assert make_examples_from_texts(("", "tiny"), tokenizer, 64) == []
