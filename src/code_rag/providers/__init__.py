"""Concrete provider implementations of the contracts in :mod:`code_rag.protocols`."""

from .fake import FakeEmbeddings, FakeLLM, InMemoryVectorStore
from .openrouter import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    OpenRouterEmbeddings,
    OpenRouterLLM,
)
from .qdrant import QdrantVectorStore

__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "OPENROUTER_BASE_URL",
    "OpenRouterEmbeddings",
    "OpenRouterLLM",
    "QdrantVectorStore",
    "FakeEmbeddings",
    "FakeLLM",
    "InMemoryVectorStore",
]
