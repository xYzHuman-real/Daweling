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
- resumable optimizer/model checkpoints
- validation-loss measurement
- benchmark and regression evaluation
- checkpoint selection

The default small experiment is intentionally modest: 4 Transformer blocks, 4 attention heads, hidden size 256, context length 256, and a small byte/special-token vocabulary.

This is **not a frontier model**. It is the foundation for building Daweling's own training, evaluation, data, and inference stack.

## Reproducible training pipeline

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
checkpoint + optimizer state
    ↓
resumable run manifest
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

The trainer periodically evaluates validation loss and writes the latest resumable checkpoint. Whenever validation improves, it also writes a `.best.pt` checkpoint. The best checkpoint is selected by lowest validation loss rather than final training loss.

### Resume training

A checkpoint contains the model state, AdamW optimizer state, completed step, run ID, seed, dataset fingerprint, configuration, and best-validation state.

Resume to a larger target step count with:

```bash
python -m training.train corpus.txt --output data/daweling-small.pt --steps 1000 --resume-from data/daweling-small.pt --seed 0
```

The trainer rejects a resume when the model configuration or deterministic run identity does not match. This prevents silently continuing an experiment with incompatible settings.

The sidecar manifest (`.manifest.json`) records the final step, best validation loss, best step, checkpoint fingerprint, and parent checkpoint when a run is resumed.

## Validation

Validation loss is computed with `training.train.validation_loss` using token-weighted mean cross-entropy and no gradient updates. Keeping validation separate from optimization makes overfitting visible and gives checkpoint selection a reproducible objective.

## Experiment lineage

`training.experiment.TrainingRunManifest` provides a machine-readable link between:

```text
dataset → configuration → seed → checkpoint → resume history
```

Run IDs are derived deterministically from the stage, dataset fingerprint, model configuration, training configuration, and seed. This makes repeated experiments comparable and makes accidental configuration drift visible.

## Evaluation and release

Evaluation records can be stored independently from checkpoints. The evaluation layer supports benchmark suites, regression checks, experiment history, and policy-driven checkpoint selection. A release candidate should be accepted only when its evaluation passes the configured regression policy.

## Data and model ownership

External model providers may be used temporarily for product/runtime development, but they are not the long-term definition of Daweling. The long-term objective is to improve Daweling-owned data, training, evaluation, inference, and model capabilities as compute and research resources grow.
