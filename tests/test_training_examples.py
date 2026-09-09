from model import DawelingTokenizer
from training.train import make_examples_from_texts


def test_dataset_examples_do_not_cross_boundaries():
    tokenizer = DawelingTokenizer()
    sequence_length = 4

    # Each source is exactly four UTF-8 bytes. With BOS/EOS, neither source
    # contains enough tokens for a full window; joining them would incorrectly
    # manufacture a window spanning the two sources.
    assert make_examples_from_texts(("abcd", "WXYZ"), tokenizer, sequence_length) == []


def test_empty_or_short_dataset_examples_are_skipped():
    tokenizer = DawelingTokenizer()
    assert make_examples_from_texts(("", "tiny"), tokenizer, 64) == []
