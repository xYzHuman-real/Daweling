# Daweling Model Foundation

Daweling owns a small decoder-only Transformer foundation model rather than treating an external provider as its identity.

## Current model

The repository contains a dependency-light PyTorch implementation with:

- deterministic UTF-8 byte tokenization
- decoder-only Transformer blocks
- tied input/output embeddings
- AdamW optimization
- autoregressive inference
- checkpoint save/load
- dataset validation and cleaning
- deterministic dataset splitting
- reproducible training-run manifests
- deterministic token batching
- validation-loss measurement
- benchmark and regression evaluation
- checkpoint selection

The default small experiment is intentionally modest: 4 Transformer blocks, 4 attention heads, hidden size 256, context length 256, and a small byte/special-token vocabulary.

This is **not a frontier model**. It is the foundation for building Daweling's own training, evaluation, data, and inference stack.

## Reproducible training pipeline

The intended development loop is:

```text
raw dataset
    ↓
validation + normalization + deduplication
    ↓
dataset manifest
    ↓
deterministic train/validation split
    ↓
token batching
    ↓
training
    ↓
periodic validation loss
    ↓
checkpoint + training-run manifest
    ↓
benchmark evaluation
    ↓
regression gate
    ↓
best accepted checkpoint
```

### Dataset preparation

```bash
python data/prepare.py input.jsonl cleaned.jsonl --manifest data/manifests/v1.json --version v1
```

The manifest records dataset version, input/output SHA-256 fingerprints, valid/duplicate/invalid counts, preprocessing configuration, and provenance metadata.

### Train

```bash
python -m training.train corpus.txt --output data/daweling-small.pt --steps 100 --seed 0
```

For a separate validation corpus:

```bash
python -m training.train corpus.txt --validation-text validation.txt --output data/daweling-small.pt --steps 100 --batch-size 4 --validation-interval 10
```

If a prepared dataset manifest is available, pass it with `--dataset-manifest`. The resulting checkpoint receives a sidecar training manifest containing the dataset fingerprint, model configuration, training configuration, seed, checkpoint fingerprint, and source text fingerprints.

## Validation

`training.validation.evaluate_loss` computes token-weighted mean cross-entropy over validation batches without changing model weights. Validation loss is kept separate from training loss so experiments can detect overfitting instead of selecting checkpoints only from the final training step.

## Experiment lineage

`training.experiment.TrainingRunManifest` provides a stable machine-readable link between:

```text
dataset → configuration → seed → checkpoint
```

Run IDs are derived deterministically from the stage, dataset fingerprint, model configuration, training configuration, and seed. This makes repeated experiments comparable and makes accidental configuration drift visible.

## Evaluation and release

Evaluation records can be stored independently from checkpoints. The evaluation layer supports benchmark suites, regression checks, experiment history, and policy-driven checkpoint selection. A release candidate should be accepted only when its evaluation passes the configured regression policy.

## Data and model ownership

External model providers may be used temporarily for product/runtime development, but they are not the long-term definition of Daweling. The long-term objective is to improve Daweling-owned data, training, evaluation, inference, and model capabilities as compute and research resources grow.
