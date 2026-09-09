"""Public data-engine APIs."""

from .loader import DatasetPartitions, load_partitions
from .manifest import DatasetManifest, canonical_example_hash, sha256_file
from .prepare import normalize_example, prepare_dataset, validate_example
from .quality import ContaminationMatch, QualityIssue, QualityResult, find_benchmark_contamination, find_duplicate_texts, normalize_for_quality, quality_check
from .split import read_jsonl, split_examples, write_jsonl

__all__ = [
    "ContaminationMatch",
    "DatasetManifest",
    "DatasetPartitions",
    "QualityIssue",
    "QualityResult",
    "canonical_example_hash",
    "find_benchmark_contamination",
    "find_duplicate_texts",
    "load_partitions",
    "normalize_example",
    "normalize_for_quality",
    "prepare_dataset",
    "quality_check",
    "read_jsonl",
    "sha256_file",
    "split_examples",
    "validate_example",
    "write_jsonl",
]
