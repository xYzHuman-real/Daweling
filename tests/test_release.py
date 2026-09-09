from pathlib import Path

from evaluation.benchmarks import BenchmarkCase
from evaluation.experiment import load_experiment
from evaluation.release import evaluate_and_select_checkpoints


def test_release_pipeline_evaluates_all_candidates_and_selects_best(monkeypatch, tmp_path):
    scores = {"data/a.pt": "wrong", "data/b.pt": "4"}

    def fake_generate(path, prompt, *, config, device):
        return scores[str(path)]

    monkeypatch.setattr("evaluation.checkpoint_evaluator.generate_from_checkpoint", fake_generate)

    selection = evaluate_and_select_checkpoints(
        [Path("data/a.pt"), Path("data/b.pt")],
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        experiment_dir=tmp_path,
    )

    assert selection.selected.path == Path("data/b.pt")
    assert len(selection.candidates) == 2
    assert load_experiment(tmp_path / "a.json").score == 0.0
    assert load_experiment(tmp_path / "b.json").score == 1.0


def test_release_pipeline_rejects_regression_failures(monkeypatch):
    def fake_generate(path, prompt, *, config, device):
        return "4"

    monkeypatch.setattr("evaluation.checkpoint_evaluator.generate_from_checkpoint", fake_generate)

    selection = evaluate_and_select_checkpoints(
        [Path("data/a.pt"), Path("data/b.pt")],
        [("math", [BenchmarkCase("addition", "2 + 2", "4")])],
        baseline_score=1.0,
        regression_threshold=0.0,
    )

    assert selection.selected.path == Path("data/a.pt")
    assert selection.rejected == ()
