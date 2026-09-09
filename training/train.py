"""Reproducible local pretraining loop for the first Daweling model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.manifest import DatasetManifest
from model import DawelingTokenizer, ModelConfig, DawelingTransformer
from training.experiment import TrainingRunManifest, make_run_id, sha256_file


def make_examples(text: str, tokenizer: DawelingTokenizer, sequence_length: int):
    ids = tokenizer.encode(text)
    usable = len(ids) - 1
    for start in range(0, usable - sequence_length + 1, sequence_length):
        chunk = ids[start : start + sequence_length + 1]
        yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def train(
    text_path: Path,
    output_path: Path,
    steps: int,
    learning_rate: float,
    *,
    seed: int = 0,
    dataset_manifest_path: Path | None = None,
) -> TrainingRunManifest:
    """Train a model and persist its exact data/config lineage beside the checkpoint."""
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    torch.manual_seed(seed)

    tokenizer = DawelingTokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = DawelingTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    text = text_path.read_text(encoding="utf-8")
    examples = list(make_examples(text, tokenizer, config.max_sequence_length))
    if not examples:
        raise ValueError("training text is too short for the configured sequence length")

    model.train()
    for step in range(steps):
        input_ids, targets = examples[step % len(examples)]
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids.unsqueeze(0), targets.unsqueeze(0))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step={step + 1} loss={loss.item():.4f}")

    dataset_sha256 = None
    if dataset_manifest_path is not None:
        manifest = DatasetManifest.load(dataset_manifest_path)
        dataset_sha256 = manifest.output_sha256

    training_config = {
        "steps": steps,
        "learning_rate": learning_rate,
        "sequence_length": config.max_sequence_length,
        "optimizer": "AdamW",
        "gradient_clip_norm": 1.0,
    }
    model_config = config.__dict__
    run_id = make_run_id(
        stage="pretraining",
        dataset_sha256=dataset_sha256,
        model_config=model_config,
        training_config=training_config,
        seed=seed,
    )

    # Store the lineage in the checkpoint itself as well as a human-readable sidecar.
    checkpoint = {
        "format_version": 2,
        "config": model_config,
        "state_dict": model.state_dict(),
        "stage": "pretraining",
        "run_id": run_id,
        "seed": seed,
        "training_config": training_config,
        "dataset_manifest": str(dataset_manifest_path) if dataset_manifest_path else None,
        "dataset_sha256": dataset_sha256,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    lineage = TrainingRunManifest(
        run_id=run_id,
        stage="pretraining",
        dataset_manifest=str(dataset_manifest_path) if dataset_manifest_path else None,
        dataset_sha256=dataset_sha256,
        model_config=model_config,
        training_config=training_config,
        seed=seed,
        checkpoint_path=str(output_path),
        checkpoint_sha256=sha256_file(output_path),
        metadata={"training_text_sha256": sha256_file(text_path)},
    )
    lineage.save(output_path.with_suffix(output_path.suffix + ".manifest.json"))
    print(f"saved checkpoint: {output_path}")
    print(f"saved run manifest: {output_path.with_suffix(output_path.suffix + '.manifest.json')}")
    return lineage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/daweling-small.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    args = parser.parse_args()
    train(args.text, args.output, args.steps, args.learning_rate, seed=args.seed, dataset_manifest_path=args.dataset_manifest)
