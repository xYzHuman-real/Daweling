from pathlib import Path

import torch

from model import DawelingTokenizer, ModelConfig, DawelingTransformer
from training.train import _load_resume, _save_checkpoint


def test_checkpoint_contains_rng_state_and_restores_it(tmp_path: Path):
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "checkpoint.pt"

    expected_rng = torch.get_rng_state().clone()
    _save_checkpoint(
        checkpoint,
        model,
        optimizer,
        step=3,
        run_id="run-1",
        seed=0,
        config=config.__dict__,
        training_config={"steps": 3},
        dataset_manifest=None,
        dataset_sha256="dataset-sha",
        best_validation_loss=1.0,
        best_step=3,
        parent_checkpoint=None,
    )

    torch.rand(8)
    _load_resume(checkpoint, model, optimizer)

    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_old_checkpoint_format_is_rejected(tmp_path: Path):
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "old.pt"
    torch.save({"format_version": 3, "config": config.__dict__, "state_dict": model.state_dict()}, checkpoint)

    try:
        _load_resume(checkpoint, model, optimizer)
    except ValueError as exc:
        assert "format 4" in str(exc)
    else:
        raise AssertionError("legacy checkpoint should be rejected")
