"""
In-memory test doubles for the provider contracts.

These let the ingest pipeline, enrichment, and server be tested with no network,
no API keys, and no Docker. They are intentionally simple, not fast.
"""

import hashlib
import json
import math
from typing import Any

from ..models import Chunk, point_id
from .openrouter import EMBEDDING_DIM


class FakeEmbeddings:
    """Deterministic embedding provider for tests.

    The same text always yields the same vector (seeded by a hash of the text),
    so tests are reproducible across runs and processes. Vectors are unit-norm.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        # Linear congruential generator — deterministic and dependency-free.
        vals: list[float] = []
        state = seed
        for _ in range(self._dimension):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            vals.append(state / 0x7FFFFFFF - 0.5)
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# Mirror the Qdrant payload so tests see the same fields.
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


class InMemoryVectorStore:
    """List-backed vector store with brute-force cosine search."""

    def __init__(self) -> None:
        # point_id -> {"vector": [...], "payload": {...}}
        self._points: dict[str, dict] = {}

    def ensure_collection(self, name: str, vector_size: int, recreate: bool = False) -> None:
        if recreate:
            self._points.clear()

    def upsert(
        self, collection_name: str, chunks_with_embeddings: list[tuple[Chunk, list[float]]]
    ) -> int:
        for chunk, embedding in chunks_with_embeddings:
            payload = {f: getattr(chunk, f, None) for f in _PAYLOAD_FIELDS}
            self._points[point_id(chunk)] = {"vector": list(embedding), "payload": payload}
        return len(chunks_with_embeddings)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        min_math_density: float | None = None,
        language: str | None = None,
    ) -> list[dict]:
        scored = []
        for point in self._points.values():
            payload = point["payload"]
            density = payload.get("math_density") or 0.0
            if min_math_density is not None and density < min_math_density:
                continue
            if language is not None and payload.get("language") != language:
                continue
            score = _cosine(query_vector, point["vector"])
            scored.append((score, payload))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {f: payload.get(f) for f in _PAYLOAD_FIELDS} | {"score": score}
            for score, payload in scored[:top_k]
        ]

    def fetch_parent(self, collection_name: str, chunk_id: str) -> dict | None:
        for point in self._points.values():
            if point["payload"].get("chunk_id") == chunk_id:
                return {f: point["payload"].get(f) for f in _PAYLOAD_FIELDS}
        return None

    def delete_by_source(self, collection_name: str, source_path: str) -> int:
        """Delete every point belonging to one source file. Returns count deleted."""
        doomed = [
            pid for pid, p in self._points.items() if p["payload"].get("source_path") == source_path
        ]
        for pid in doomed:
            del self._points[pid]
        return len(doomed)

    def list_source_paths(self, collection_name: str) -> set[str]:
        """Return the distinct ``source_path`` values currently stored."""
        return {
            p["payload"]["source_path"]
            for p in self._points.values()
            if p["payload"].get("source_path") is not None
        }


class FakeLLM:
    """Canned chat-completion provider.

    By default it inspects the enrichment prompt, counts the ``--- CHUNK n ---``
    delimiters, and returns a well-formed batch response with one entry per
    chunk. Pass ``response`` to force a fixed string instead (useful for testing
    parse failures and fallbacks).
    """

    def __init__(self, response: str | None = None) -> None:
        self._response = response
        self.calls: list[list[dict]] = []  # captured message lists, for assertions

    async def complete(self, messages: list[dict], model: str, **kwargs: Any) -> str:
        self.calls.append(messages)
        if self._response is not None:
            return self._response

        user_content = messages[-1]["content"] if messages else ""
        n = user_content.count("--- CHUNK ") or 1
        items = [
            {
                "id": i,
                "summary": f"Fake summary for chunk {i}.",
                "hypothetical_questions": [
                    f"What does chunk {i} do?",
                    f"When would I use chunk {i}?",
                ],
            }
            for i in range(1, n + 1)
        ]
        return json.dumps({"chunks": items})
