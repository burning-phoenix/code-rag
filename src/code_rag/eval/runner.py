"""
Eval runner: indexes the corpus under a given configuration, runs each golden
query, and collects the retrieved items for metric computation.

This module constructs and owns real providers, so it is run by hand (needs an
API key and a running Qdrant), never in CI. Metric logic lives in
:mod:`code_rag.eval.metrics` and is unit-tested separately.
"""

import logging
from dataclasses import dataclass, replace

from ..chunkers import ChunkerRegistry, get_registry
from ..config import ProjectConfig
from ..ingest import ingest
from ..pointer_index import load_index, search_index
from ..protocols import EmbeddingProvider, LLMProvider, VectorStore
from .dataset import GoldenDataset, Query
from .line_chunker import WHOLE_FILE_EMBED_CHARS, WHOLE_FILE_WINDOW, build_line_registry
from .metrics import QueryResult

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """One comparison configuration: a label plus the settings that define it."""

    label: str
    enrich: bool = True
    chunker: str = "ast"  # "ast" | "line-based" | "whole-file"
    enrichment_model: str | None = None  # None = the project config's model

    def registry(self) -> ChunkerRegistry:
        if self.chunker == "line-based":
            return build_line_registry()
        if self.chunker == "whole-file":
            return build_line_registry(WHOLE_FILE_WINDOW, max_chars=WHOLE_FILE_EMBED_CHARS)
        return get_registry()


class EvalRunner:
    """Indexes a corpus and retrieves results for a golden dataset."""

    def __init__(
        self,
        config: ProjectConfig,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider | None = None,
        enrich_batch_size: int = 10,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.enrich_batch_size = enrich_batch_size

    async def _index(self, run: RunConfig) -> None:
        llm = self.llm if run.enrich else None
        logger.info(
            "Indexing corpus for run %r (enrich=%s, chunker=%s)", run.label, run.enrich, run.chunker
        )
        # Enrichment-model comparisons vary the model per run; ingest reads it
        # from the config, so hand it a per-run copy rather than mutating ours.
        config = self.config
        if run.enrichment_model is not None:
            config = replace(config, enrichment_model=run.enrichment_model)
        await ingest(
            config,
            self.embedder,
            self.store,
            llm=llm,
            recreate_collection=True,
            enrich_batch_size=self.enrich_batch_size,
            registry=run.registry(),
        )

    async def embed_queries(
        self, dataset: GoldenDataset, batch_size: int = 10
    ) -> dict[str, list[float]]:
        """Embed every semantic query's text in batches, keyed by query id.

        Query vectors depend only on the embedding model, not on how the corpus
        was indexed — so one pass here serves every configuration in a matrix
        run instead of one embed call per query per config. ``lookup_index``
        queries are skipped (that tool never embeds).
        """
        semantic = [q for q in dataset.queries if q.tool != "lookup_index"]
        vectors: dict[str, list[float]] = {}
        for i in range(0, len(semantic), batch_size):
            batch = semantic[i : i + batch_size]
            embedded = await self.embedder.embed([q.query for q in batch])
            vectors.update({q.id: v for q, v in zip(batch, embedded, strict=True)})
        logger.info(
            "Embedded %d queries in %d batched calls (reused across all configs)",
            len(semantic),
            -(-len(semantic) // batch_size),
        )
        return vectors

    async def _retrieve(
        self, query: Query, top_k: int, query_vector: list[float] | None
    ) -> list[dict]:
        """Retrieve top-k items for a query, normalised to span-matchable dicts."""
        if query.tool == "lookup_index":
            index = load_index(self.config.pointer_index)
            entries = search_index(index, query.query)[:top_k]
            return [
                {
                    "file_name": e["file_name"],
                    "start_line": e["start_line"],
                    "end_line": e["end_line"],
                    "symbol_name": e["concept"],
                }
                for e in entries
            ]

        if query_vector is None:
            query_vector = (await self.embedder.embed([query.query]))[0]
        return self.store.search(self.config.collection, query_vector, top_k=top_k)

    async def evaluate(
        self,
        dataset: GoldenDataset,
        run: RunConfig,
        max_top_k: int,
        query_vectors: dict[str, list[float]] | None = None,
    ) -> list[QueryResult]:
        """Index under ``run`` then retrieve ``max_top_k`` items per query.

        Pass ``query_vectors`` (from :meth:`embed_queries`) to reuse
        pre-computed query embeddings; without it each query embeds one call
        at a time.
        """
        await self._index(run)
        vectors = query_vectors or {}
        results: list[QueryResult] = []
        for query in dataset.queries:
            retrieved = await self._retrieve(query, max_top_k, vectors.get(query.id))
            results.append(QueryResult(query=query, retrieved=retrieved))
        return results
