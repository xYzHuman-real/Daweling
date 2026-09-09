from model.tokenizer import DawelingTokenizer


def test_tokenizer_round_trip_utf8():
    tokenizer = DawelingTokenizer()
    text = "Daweling — नमस्ते 🌍"
    ids = tokenizer.encode(text)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text


def test_tokenizer_special_tokens_are_reserved():
    tokenizer = DawelingTokenizer()
    assert tokenizer.vocab_size == 260
    assert tokenizer.pad_id == 256
    assert tokenizer.bos_id == 257
    assert tokenizer.eos_id == 258
    assert tokenizer.unk_id == 259


def test_tokenizer_can_omit_special_tokens():
    tokenizer = DawelingTokenizer()
    assert tokenizer.encode("abc", add_bos=False, add_eos=False) == [97, 98, 99]
