import torch

from model.generate import GenerationConfig, generate
from model.tokenizer import DawelingTokenizer


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.config = type("Config", (), {"max_seq_len": 32})()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, tokens):
        logits = torch.full(
            (tokens.size(0), tokens.size(1), self.vocab_size),
            -100.0,
            device=tokens.device,
        )
        logits[:, :, ord("!")] = 100.0
        return logits


def test_generation_is_deterministic_without_sampling():
    tokenizer = DawelingTokenizer()
    model = DummyModel(tokenizer.vocab_size)
    text = generate(
        model,
        tokenizer,
        "Hi",
        GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    assert text.endswith("!!")
