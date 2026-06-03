"""
Qdrant-backed vector store.

Wraps a :class:`qdrant_client.QdrantClient` and exposes the
:class:`~code_rag.protocols.VectorStore` contract. Supports extended payloads
for code chunks (language, symbol info, parent-child links).
"""

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from ..models import Chunk, point_id

logger = logging.getLogger(__name__)

# All payload fields stored per point and returned from search results.
_PAYLOAD_FIELDS = [
    "text",
    "file_name",
    "source_path",
    "section_title",
    "start_line",
    "end_line",
    "math_density",
    "chunk_id",
    "chunk_type",
    "language",
    "symbol_name",
    "symbol_type",
    "parent_id",
    "summary",
]


class QdrantVectorStore:
    """Vector store backed by a Qdrant collection."""

    def __init__(self, url: str) -> None:
        self._client = QdrantClient(url=url)

    def ensure_collection(self, name: str, vector_size: int, recreate: bool = False) -> None:
        """Ensure the collection exists, optionally dropping and recreating it."""
        collections = [c.name for c in self._client.get_collections().collections]

        if name in collections:
            if recreate:
                logger.info("Deleting existing collection '%s'", name)
                self._client.delete_collection(name)
            else:
                logger.info("Collection '%s' already exists", name)
                return

        logger.info("Creating collection '%s' (dim=%d, distance=Cosine)", name, vector_size)
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(
        self,
        collection_name: str,
        chunks_with_embeddings: list[tuple[Chunk, list[float]]],
        batch_size: int = 50,
    ) -> int:
        """Upsert (Chunk, embedding) pairs; return the number of points written.

        Point IDs are derived deterministically from chunk identity so repeated
        ingests deduplicate rather than accumulate.
        """
        points = []
        for chunk, embedding in chunks_with_embeddings:
            points.append(
                PointStruct(
                    id=point_id(chunk),
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "file_name": chunk.file_name,
                        "source_path": chunk.source_path,
                        "section_title": chunk.section_title,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "math_density": chunk.math_density,
                        "chunk_id": chunk.chunk_id,
                        "chunk_type": chunk.chunk_type,
                        "language": chunk.language,
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "parent_id": chunk.parent_id,
                        "summary": chunk.summary,
                    },
                )
            )

        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(collection_name=collection_name, wait=True, points=batch)
            total += len(batch)
            logger.info("Upserted batch %d–%d / %d", i + 1, i + len(batch), len(points))

        return total

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        min_math_density: float | None = None,
        language: str | None = None,
    ) -> list[dict]:
        """Search for chunks similar to ``query_vector`` with optional filters."""
        conditions: list = []
        if min_math_density is not None:
            conditions.append(FieldCondition(key="math_density", range=Range(gte=min_math_density)))
        if language is not None:
            conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))

        query_filter = Filter(must=conditions) if conditions else None

        results = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {field: (r.payload or {}).get(field) for field in _PAYLOAD_FIELDS} | {"score": r.score}
            for r in results
        ]

    def fetch_parent(self, collection_name: str, chunk_id: str) -> dict | None:
        """Fetch a single chunk payload by ``chunk_id`` (for parent-child context)."""
        results, _ = self._client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))]
            ),
            limit=1,
            with_payload=True,
        )

        if results:
            payload = results[0].payload or {}
            return {field: payload.get(field) for field in _PAYLOAD_FIELDS}
        return None

    def delete_by_source(self, collection_name: str, source_path: str) -> int:
        """Delete every point belonging to one source file. Returns count deleted."""
        condition = Filter(
            must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))]
        )
        # Count first so we can report what we purged (delete returns no count).
        n = self._client.count(
            collection_name=collection_name, count_filter=condition, exact=True
        ).count
        if n:
            self._client.delete(
                collection_name=collection_name, points_selector=condition, wait=True
            )
        return n

    def list_source_paths(self, collection_name: str) -> set[str]:
        """Return the distinct ``source_path`` values currently stored."""
        paths: set[str] = set()
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=["source_path"],
                with_vectors=False,
            )
            for rec in records:
                src = (rec.payload or {}).get("source_path")
                if src is not None:
                    paths.add(src)
            if offset is None:
                break
        return paths
