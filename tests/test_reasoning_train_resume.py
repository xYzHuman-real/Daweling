from pathlib import Path

import pytest

from training.reasoning_train import _load_resume, _training_config


def test_resume_rejects_dataset_mismatch(tmp_path: Path):
    # The full training loop creates checkpoints; this test exercises the safety gate
    # against accidentally resuming a run with different data/configuration.
    import torch
    from model import DawelingTokenizer, DawelingTransformer, ModelConfig

    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = tmp_path / "resume.pt"
    torch.save({
        "format_version": 1,
        "config": config.__dict__,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_id": "run",
        "dataset_sha256": "old",
        "training_config": _training_config(10, 1e-4, 0.1, 2, 0, 0.0),
        "seed": 0,
        "step": 2,
    }, checkpoint)

    with pytest.raises(ValueError, match="dataset"):
        _load_resume(
            checkpoint, model, optimizer, device="cpu", expected_run_id="run",
            dataset_sha256="new", training_config=_training_config(20, 1e-4, 0.1, 2, 0, 0.0), seed=0,
        )
