from pathlib import Path

import torch

from evaluation.reasoning import ReasoningCase
from evaluation.reasoning_checkpoint import evaluate_reasoning_checkpoint, load_reasoning_checkpoint
from model import DawelingTokenizer, DawelingTransformer, ModelConfig


def _checkpoint(path: Path) -> None:
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, path)


def test_reasoning_checkpoint_loads(tmp_path: Path):
    path = tmp_path / "reasoning.pt"
    _checkpoint(path)
    assert isinstance(load_reasoning_checkpoint(path), DawelingTransformer)


def test_reasoning_checkpoint_evaluation_uses_callback(tmp_path: Path):
    path = tmp_path / "reasoning.pt"
    _checkpoint(path)
    cases = [ReasoningCase("one", "1+1", "2", "math")]
    calls = []

    def predict(model, tokenizer, prompt):
        calls.append((model.training, prompt, tokenizer.vocab_size))
        return "2"

    report = evaluate_reasoning_checkpoint(path, cases, predict)
    assert report.score == 1.0
    assert calls == [(False, "1+1", DawelingTokenizer().vocab_size)]
