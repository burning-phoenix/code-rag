"""
Qdrant indexer: manages the vector collection and upserts embedded chunks.
Supports extended payloads for code chunks (language, symbol info, parent-child links).
"""

import hashlib
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
    MatchValue,
)

from .embeddings import EMBEDDING_DIM

logger = logging.getLogger(__name__)

# All payload fields returned from search results
_PAYLOAD_FIELDS = [
    "text", "file_name", "section_title", "start_line", "end_line",
    "math_density", "chunk_id", "chunk_type", "language",
    "symbol_name", "symbol_type", "parent_id", "summary",
]


def get_client(url: str) -> QdrantClient:
    """Create a Qdrant client."""
    return QdrantClient(url=url)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = EMBEDDING_DIM,
    recreate: bool = False,
) -> None:
    """
    Ensure the Qdrant collection exists. Optionally recreate it.
    """
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        if recreate:
            logger.info("Deleting existing collection '%s'", collection_name)
            client.delete_collection(collection_name)
        else:
            logger.info("Collection '%s' already exists", collection_name)
            return

    logger.info(
        "Creating collection '%s' (dim=%d, distance=Cosine)",
        collection_name,
        vector_size,
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def upsert_chunks(
    client: QdrantClient,
    chunks_with_embeddings: list[tuple],
    collection_name: str,
    batch_size: int = 50,
) -> int:
    """
    Upsert chunks with their embeddings into Qdrant.

    Args:
        client: Qdrant client instance.
        chunks_with_embeddings: List of (Chunk, embedding_vector) tuples.
        collection_name: Target collection name.
        batch_size: Number of points per upsert batch.

    Returns:
        Number of points upserted.
    """
    points = []
    for chunk, embedding in chunks_with_embeddings:
        # Deterministic ID from chunk identity so upserts deduplicate across runs
        identity = f"{chunk.file_name}:{chunk.chunk_type}:{chunk.start_line}-{chunk.end_line}"
        point_id = str(uuid.UUID(hashlib.md5(identity.encode()).hexdigest()))
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "text": chunk.text,
                "file_name": chunk.file_name,
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
        points.append(point)

    # Upsert in batches
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=batch,
        )
        total += len(batch)
        logger.info("Upserted batch %d–%d / %d", i + 1, i + len(batch), len(points))

    return total


def search(
    client: QdrantClient,
    query_vector: list[float],
    collection_name: str,
    top_k: int = 5,
    min_math_density: float | None = None,
    language: str | None = None,
) -> list[dict]:
    """
    Search Qdrant for chunks similar to the query vector.

    Supports optional filters:
    - min_math_density: filter for math-heavy content
    - language: filter by programming language
    """
    conditions = []
    if min_math_density is not None:
        conditions.append(
            FieldCondition(key="math_density", range=Range(gte=min_math_density))
        )
    if language is not None:
        conditions.append(
            FieldCondition(key="language", match=MatchValue(value=language))
        )

    query_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {field: r.payload.get(field) for field in _PAYLOAD_FIELDS} | {"score": r.score}
        for r in results
    ]


def fetch_parent(
    client: QdrantClient,
    collection_name: str,
    chunk_id: str,
) -> dict | None:
    """
    Fetch a single chunk by its chunk_id (used for parent-child retrieval).

    Returns the payload dict or None if not found.
    """
    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))
            ]
        ),
        limit=1,
        with_payload=True,
    )

    if results:
        return {field: results[0].payload.get(field) for field in _PAYLOAD_FIELDS}
    return None
