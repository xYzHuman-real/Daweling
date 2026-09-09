# Daweling Data Engine

The data engine turns raw JSONL examples into reproducible training inputs.

## Pipeline

`raw JSONL → validation → normalization → deterministic deduplication → fingerprinted dataset → manifest`

Each prepared dataset can have a JSON manifest containing:

- dataset version
- input/output paths
- SHA-256 fingerprints
- valid, duplicate, and invalid counts
- preprocessing configuration
- source/provenance metadata

## Prepare a dataset

The original two-argument command remains supported:

```bash
python data/prepare.py input.jsonl cleaned.jsonl
```

To also write a manifest:

```bash
python data/prepare.py input.jsonl cleaned.jsonl --manifest data/manifests/v1.json --version v1 --source my-source
```

Invalid JSON lines and schema-invalid examples are excluded rather than entering training data. Duplicate examples are removed deterministically using a canonical SHA-256 fingerprint.

## Reproducibility

A manifest should travel with the dataset through training and evaluation. The input and output hashes make it possible to detect accidental changes before an experiment is reproduced.

Large corpora should remain outside the repository. Never commit secrets, private personal data, or material you do not have permission to use.
