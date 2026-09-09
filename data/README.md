# Daweling Data

This directory contains the reproducible data pipeline used to prepare datasets for Daweling model development.

## Pipeline

```text
raw JSONL
   ↓
validation
   ↓
normalization
   ↓
deduplication
   ↓
clean JSONL + dataset manifest
   ↓
deterministic train/validation split
   ↓
training
```

## Dataset manifest

`data.prepare.prepare_dataset()` creates a `DatasetManifest` containing:

- dataset version
- input and output SHA-256 fingerprints
- example count
- duplicate count
- source inventory
- preprocessing configuration
- optional experiment metadata

Example:

```bash
python data/prepare.py raw.jsonl data/clean.jsonl --version train-v1 --manifest data/train-v1.manifest.json
```

The manifest makes a prepared dataset identifiable and helps connect future training runs to the exact data artifact used.

## Data quality

Each example must contain a non-empty `text` and `source`. Optional `quality` values must be numeric and between 0 and 1. Preparation trims the required string fields and removes exact canonical duplicates deterministically.

## Principles

- Keep raw/source data separate from generated and processed data.
- Never commit secrets, private personal data, or licensed material without permission.
- Record dataset provenance and transformations in metadata.
- Keep evaluation data separate from training data to reduce leakage.
- Do not commit large training corpora or generated dataset artifacts to the repository.
