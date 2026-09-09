from pathlib import Path

import torch

from evaluation.reasoning import ReasoningCase
from evaluation.reasoning_regression import compare_reasoning_checkpoints
from model import DawelingTokenizer, DawelingTransformer, ModelConfig


def _checkpoint(path: Path) -> None:
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, path)


def test_reasoning_regression_passes_when_current_matches_baseline(tmp_path: Path):
    baseline = tmp_path / "baseline.pt"
    current = tmp_path / "current.pt"
    _checkpoint(baseline)
    _checkpoint(current)
    cases = [ReasoningCase("one", "1+1", "2")]

    def predict(model, tokenizer, prompt):
        return "2"

    report = compare_reasoning_checkpoints(baseline, current, cases, predict)
    assert report.passed
    assert report.regression.delta == 0.0


def test_reasoning_regression_rejects_lower_current_score(tmp_path: Path):
    baseline = tmp_path / "baseline.pt"
    current = tmp_path / "current.pt"
    _checkpoint(baseline)
    _checkpoint(current)
    cases = [
        ReasoningCase("one", "1+1", "2"),
        ReasoningCase("two", "2+2", "4"),
    ]
    calls = {"count": 0}

    def predict(model, tokenizer, prompt):
        calls["count"] += 1
        return "2" if calls["count"] <= 2 else "wrong"

    report = compare_reasoning_checkpoints(baseline, current, cases, predict)
    assert not report.passed
    assert report.regression.delta < 0
