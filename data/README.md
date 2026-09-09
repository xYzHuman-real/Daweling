# Daweling Data Engine

The data engine turns raw JSONL examples into reproducible, quality-controlled training inputs.

## Pipeline

`raw JSONL → schema validation → quality filtering → normalization → contamination filtering → deterministic deduplication → fingerprinted dataset → manifest`

Each prepared dataset can have a JSON manifest containing:

- dataset version
- input/output paths
- SHA-256 fingerprints
- valid, duplicate, invalid, and quality-rejected counts
- quality issue counts
- contamination-rejected counts and benchmark IDs
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

### Quality controls

Examples are rejected when they fail schema validation or configured quality limits. By default, examples must contain between 1 and 200,000 characters. The limits can be tightened for a particular corpus:

```bash
python data/prepare.py input.jsonl cleaned.jsonl --min-chars 20 --max-chars 100000
```

Accepted text is stripped of surrounding whitespace before fingerprinting and output. Deduplication uses a canonical SHA-256 representation, so the same normalized example is emitted only once and the result is deterministic.

### Evaluation contamination controls

A benchmark JSONL file can be supplied to exclude training examples containing the normalized benchmark prompt + expected answer pair:

```bash
python data/prepare.py input.jsonl cleaned.jsonl --benchmark-jsonl evaluation.jsonl
```

The manifest records which benchmark IDs caused exclusions. This makes the training/evaluation boundary auditable instead of silently relying on manual dataset inspection.

## Reproducibility

A manifest should travel with the dataset through training and evaluation. The input and output hashes make it possible to detect accidental changes before an experiment is reproduced.

Large corpora should remain outside the repository. Never commit secrets, private personal data, or material you do not have permission to use.
