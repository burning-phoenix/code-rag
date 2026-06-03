"""
Abstractions the orchestrators depend on (Dependency Inversion).

``ingest``, ``server``, and ``enrichment`` are written against these contracts,
never against concrete OpenRouter/Qdrant clients. Concrete implementations live
in :mod:`code_rag.providers`; test doubles live there too.

The service contracts are :class:`typing.Protocol` (structural): a class is a
valid provider if it has the right methods — no inheritance required. The chunker
contract is an :class:`abc.ABC` instead, because chunkers are explicitly
registered and we want the interface enforced at definition time.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .config import ProjectConfig
from .models import Chunk, IndexEntry


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors for indexing and querying."""

    @property
    def dimension(self) -> int:
        """Vector dimensionality this provider produces (the collection size)."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input in order."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """A vector collection that supports indexing and similarity search.

    The four operations are cohesive — all act on the same collection resource —
    so they share one interface rather than being split per ISP.
    """

    def ensure_collection(self, name: str, vector_size: int, recreate: bool = False) -> None:
        """Create the collection if missing (optionally drop and recreate)."""
        ...

    def upsert(
        self, collection_name: str, chunks_with_embeddings: list[tuple[Chunk, list[float]]]
    ) -> int:
        """Upsert (Chunk, embedding) pairs; return the number of points written."""
        ...

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        min_math_density: float | None = None,
        language: str | None = None,
    ) -> list[dict]:
        """Return the top-k most similar chunks as payload dicts (with ``score``)."""
        ...

    def fetch_parent(self, collection_name: str, chunk_id: str) -> dict | None:
        """Fetch a single chunk payload by ``chunk_id`` (for parent context)."""
        ...

    def delete_by_source(self, collection_name: str, source_path: str) -> int:
        """Delete every point for one source file; return the count removed."""
        ...

    def list_source_paths(self, collection_name: str) -> set[str]:
        """Return the distinct ``source_path`` values currently stored."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """A chat-completion backend used for enrichment."""

    async def complete(self, messages: list[dict], model: str, **kwargs: Any) -> str:
        """Run a chat completion and return the assistant message content."""
        ...


class BaseChunker(ABC):
    """Contract for all chunkers. Each chunker declares its supported extensions."""

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return the file extensions this chunker handles (e.g. ``{'.md'}``)."""
        ...

    @abstractmethod
    def chunk(self, file_path: Path, config: ProjectConfig) -> list[Chunk]:
        """Chunk a single file into a list of :class:`~code_rag.models.Chunk`."""
        ...

    def scan(self, file_path: Path, config: ProjectConfig) -> list[IndexEntry]:
        """Generate pointer-index entries for a file. Defaults to none."""
        return []
