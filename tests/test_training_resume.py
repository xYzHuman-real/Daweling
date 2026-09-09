from pathlib import Path

import torch

from model import DawelingTokenizer, ModelConfig, DawelingTransformer
from training.train import _load_resume, _save_checkpoint, make_examples


def test_checkpoint_round_trip_restores_optimizer_and_step(tmp_path):
    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=1, max_sequence_length=16)
    model = DawelingTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs, targets = next(iter(make_examples("hello world " * 50, tokenizer, 16)))
    _, loss = model(inputs.unsqueeze(0), targets.unsqueeze(0))
    assert loss is not None
    loss.backward()
    optimizer.step()

    path = tmp_path / "resume.pt"
    _save_checkpoint(path, model, optimizer, step=7, run_id="run", seed=0, config=config.__dict__, training_config={"steps": 10}, dataset_manifest=None, dataset_sha256=None, best_validation_loss=1.5, best_step=7, parent_checkpoint=None)

    restored_model = DawelingTransformer(config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    checkpoint = _load_resume(path, restored_model, restored_optimizer)
    assert checkpoint["step"] == 7
    assert checkpoint["best_step"] == 7
    assert checkpoint["best_validation_loss"] == 1.5
    assert restored_optimizer.state_dict()["state"]
