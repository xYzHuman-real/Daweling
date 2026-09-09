"""Dataset preparation and provenance utilities for Daweling."""

from .manifest import DatasetManifest, canonical_example_hash, sha256_file
from .prepare import normalize_example, prepare_dataset, validate_example
from .split import split_examples

__all__ = [
    "DatasetManifest",
    "canonical_example_hash",
    "normalize_example",
    "prepare_dataset",
    "sha256_file",
    "split_examples",
    "validate_example",
]
