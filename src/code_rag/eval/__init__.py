"""Retrieval evaluation framework (diagnostic; not run in CI)."""

from .dataset import GoldenDataset, Query, Span, load_dataset
from .metrics import QueryResult, hit_rate, mrr_at_k, ndcg_at_k, recall_at_k

__all__ = [
    "GoldenDataset",
    "Query",
    "Span",
    "load_dataset",
    "QueryResult",
    "hit_rate",
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
]
