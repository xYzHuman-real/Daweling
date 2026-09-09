from pathlib import Path

from evaluation.benchmarks import BenchmarkCase
from evaluation.checkpoint_evaluator import evaluate_checkpoint


def test_evaluate_checkpoint_builds_and_persists_experiment(monkeypatch, tmp_path):
    def fake_generate(path, prompt, *, config, device):
        return {"data/a.pt": "4"}[str(path)]

    monkeypatch.setattr("evaluation.checkpoint_evaluator.generate_from_checkpoint", fake_generate)

    record = evaluate_checkpoint(
        Path("data/a.pt"),
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        metadata={"model": "daweling-small"},
        output_path=tmp_path / "a.json",
    )

    assert record.name == "a"
    assert record.score == 1.0
    assert record.metadata["checkpoint"] == "data/a.pt"
    assert record.metadata["model"] == "daweling-small"
    assert (tmp_path / "a.json").exists()
