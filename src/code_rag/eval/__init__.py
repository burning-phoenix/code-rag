"""Retrieval evaluation framework (diagnostic; not run in CI)."""

from .dataset import GoldenDataset, Query, Span, load_dataset
from .metrics import (
    QueryResult,
    concentration_at_k,
    coverage_at_k,
    load_corpus_weights,
    wholeness_at_k,
)

__all__ = [
    "GoldenDataset",
    "Query",
    "Span",
    "load_dataset",
    "QueryResult",
    "coverage_at_k",
    "concentration_at_k",
    "wholeness_at_k",
    "load_corpus_weights",
]
