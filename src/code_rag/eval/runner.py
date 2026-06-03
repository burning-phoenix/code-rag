"""
Eval runner: indexes the corpus under a given configuration, runs each golden
query, and collects the retrieved items for metric computation.

This is a composition root — it owns real providers — so it is exercised
manually (needs API keys + Qdrant), never in CI. Metric logic lives in
:mod:`code_rag.eval.metrics` and is unit-tested separately.
"""

import logging
from dataclasses import dataclass

from ..chunkers import ChunkerRegistry, get_registry
from ..config import ProjectConfig
from ..ingest import ingest
from ..pointer_index import load_index, search_index
from ..protocols import EmbeddingProvider, LLMProvider, VectorStore
from .dataset import GoldenDataset, Query
from .line_chunker import build_line_registry
from .metrics import QueryResult

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """One ablation cell: a label plus the knobs that define it."""

    label: str
    enrich: bool = True
    chunker: str = "ast"  # "ast" | "line-based"

    def registry(self) -> ChunkerRegistry:
        return build_line_registry() if self.chunker == "line-based" else get_registry()


class EvalRunner:
    """Indexes a corpus and retrieves results for a golden dataset."""

    def __init__(
        self,
        config: ProjectConfig,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider | None = None,
        enrich_batch_size: int = 5,
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
        await ingest(
            self.config,
            self.embedder,
            self.store,
            llm=llm,
            recreate_collection=True,
            enrich_batch_size=self.enrich_batch_size,
            registry=run.registry(),
        )

    async def _retrieve(self, query: Query, top_k: int) -> list[dict]:
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

        query_vector = (await self.embedder.embed([query.query]))[0]
        return self.store.search(self.config.collection, query_vector, top_k=top_k)

    async def evaluate(
        self, dataset: GoldenDataset, run: RunConfig, max_top_k: int
    ) -> list[QueryResult]:
        """Index under ``run`` then retrieve ``max_top_k`` items per query."""
        await self._index(run)
        results: list[QueryResult] = []
        for query in dataset.queries:
            retrieved = await self._retrieve(query, max_top_k)
            results.append(QueryResult(query=query, retrieved=retrieved))
        return results
