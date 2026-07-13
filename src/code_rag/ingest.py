"""
Ingest pipeline: chunks all files (markdown, code, plaintext), optionally
enriches with LLM metadata, embeds them, and indexes in Qdrant.
Also generates the static pointer index.
"""

import asyncio
import logging
import sys

from .chunkers import ChunkerRegistry, chunk_directory, get_registry
from .config import ProjectConfig
from .enrichment import enrich_chunks, get_embedding_text
from .models import Chunk
from .pointer_index import save_index
from .protocols import EmbeddingProvider, LLMProvider, VectorStore

logger = logging.getLogger(__name__)


def _sync_incremental(
    store: VectorStore,
    collection: str,
    chunks_with_embeddings: list[tuple[Chunk, list[float]]],
) -> int:
    """Make ``collection`` reflect exactly the current corpus, in place.

    The decision of *what* to sync lives here in the pipeline, not in the
    store: the store only knows how to delete and list points. For each source
    file present in this ingest, its existing points are deleted before the
    new ones are inserted (so chunks whose line ranges moved after an edit
    don't survive as stale entries); then every stored source path no longer
    present on disk is removed (so deleted files disappear from the index
    too).
    """
    by_source: dict[str, list[tuple[Chunk, list[float]]]] = {}
    for chunk, embedding in chunks_with_embeddings:
        # source_path is stamped by chunk_directory; fall back to file_name if not.
        key = chunk.source_path or chunk.file_name
        by_source.setdefault(key, []).append((chunk, embedding))

    present = set(by_source)

    # Remove files deleted from the corpus since the last ingest.
    for stale in store.list_source_paths(collection) - present:
        removed = store.delete_by_source(collection, stale)
        logger.info("Removed %d stale chunks from deleted file %s", removed, stale)

    # Replace each present file's chunks wholesale.
    indexed = 0
    for source_path, items in by_source.items():
        store.delete_by_source(collection, source_path)
        indexed += store.upsert(collection, items)
    return indexed


async def ingest(
    config: ProjectConfig,
    embedder: EmbeddingProvider,
    store: VectorStore,
    llm: LLMProvider | None = None,
    recreate_collection: bool = True,
    embedding_batch_size: int = 10,
    enrich_batch_size: int = 10,
    registry: ChunkerRegistry | None = None,
) -> dict:
    """
    Run the full ingest pipeline:
    1. Chunk all files (markdown, code, plaintext)
    2. Generate static pointer index
    3. (Optional) Enrich chunks with LLM summaries — only when ``llm`` is given
    4. Embed all chunks via the embedding provider
    5. Index embeddings in the vector store

    The pipeline depends only on the interfaces passed in, so it can be driven
    by real providers (CLI) or fakes (tests) with no code changes. Pass a
    custom ``registry`` to swap chunking strategies (the evaluation's
    comparison runs do this).

    Returns summary dict with counts.
    """
    data_dir = config.data_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if registry is None:
        registry = get_registry()

    # Step 1: Chunk all files
    logger.info(
        "Chunking files in %s (extensions: %s) ...",
        data_dir,
        ", ".join(config.file_extensions),
    )
    chunks = chunk_directory(data_dir, config, registry)

    code_chunks = sum(1 for c in chunks if c.chunk_type == "code")
    md_chunks = sum(1 for c in chunks if c.chunk_type == "markdown")
    pt_chunks = sum(1 for c in chunks if c.chunk_type == "plaintext")
    logger.info(
        "Created %d chunks (%d code, %d markdown, %d plaintext)",
        len(chunks),
        code_chunks,
        md_chunks,
        pt_chunks,
    )

    if not chunks:
        logger.warning(
            "No chunks created. Check that %s contains files with extensions: %s",
            data_dir,
            ", ".join(config.file_extensions),
        )
        return {"chunks": 0, "indexed": 0, "pointer_entries": 0}

    # Step 2: Generate static pointer index
    logger.info("Building static pointer index → %s", config.pointer_index)
    pointer_count = save_index(data_dir, config.pointer_index, config, registry)
    logger.info("Indexed %d pointer entries", pointer_count)

    # Step 3: Optional enrichment (only when an LLM provider was injected)
    if llm is not None:
        logger.info(
            "Enriching %d chunks with LLM summaries (model: %s) ...",
            len(chunks),
            config.enrichment_model,
        )
        chunks = await enrich_chunks(
            chunks,
            llm,
            model=config.enrichment_model,
            batch_size=enrich_batch_size,
        )

    # Step 4: Embed chunks in batches (using enriched text when available)
    logger.info("Embedding %d chunks (batch size: %d) ...", len(chunks), embedding_batch_size)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(chunks), embedding_batch_size):
        batch = chunks[i : i + embedding_batch_size]
        texts = [get_embedding_text(c) for c in batch]
        try:
            embeddings = await embedder.embed(texts)
        except Exception as e:
            logger.error(
                "Embedding failed at batch %d-%d / %d after retries: %s",
                i,
                min(i + embedding_batch_size, len(chunks)),
                len(chunks),
                e,
            )
            raise
        all_embeddings.extend(embeddings)
        logger.info(
            "Embedded %d / %d chunks",
            min(i + embedding_batch_size, len(chunks)),
            len(chunks),
        )

    # Step 5: Index in the vector store
    logger.info("Indexing %d chunks in collection '%s' ...", len(chunks), config.collection)
    store.ensure_collection(
        config.collection,
        vector_size=embedder.dimension,
        recreate=recreate_collection,
    )

    chunks_with_embeddings = list(zip(chunks, all_embeddings, strict=True))

    if recreate_collection:
        # Collection was just dropped and rebuilt — a plain upsert is the full state.
        indexed = store.upsert(config.collection, chunks_with_embeddings)
    else:
        # Incremental sync: make the collection reflect exactly the current corpus.
        # Deleting each present file's points before re-inserting clears chunks
        # whose line ranges moved after an edit; removing paths no longer present
        # clears the chunks of deleted files.
        indexed = _sync_incremental(store, config.collection, chunks_with_embeddings)
    logger.info("Indexed %d chunks", indexed)

    return {
        "chunks": len(chunks),
        "indexed": indexed,
        "pointer_entries": pointer_count,
        "code_chunks": code_chunks,
        "markdown_chunks": md_chunks,
        "plaintext_chunks": pt_chunks,
    }


def run_ingest(
    project_dir: str | None = None,
    no_recreate: bool = False,
    batch_size: int = 10,
    min_chunk_lines: int | None = None,
    max_chunk_lines: int | None = None,
    enrich: bool | None = None,
    enrich_batch_size: int | None = None,
) -> None:
    """CLI entry point: builds the real providers and runs the ingest pipeline."""
    from .config import load_config
    from .providers import OpenRouterEmbeddings, OpenRouterLLM, QdrantVectorStore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = load_config(project_dir)

    # CLI args override config values
    if min_chunk_lines is not None:
        config.min_chunk_lines = min_chunk_lines
    if max_chunk_lines is not None:
        config.max_chunk_lines = max_chunk_lines
    if enrich is not None:
        config.enrich_chunks = enrich

    embedder = OpenRouterEmbeddings(api_key=config.openrouter_api_key)
    store = QdrantVectorStore(config.qdrant_url)

    llm: OpenRouterLLM | None = None
    if config.enrich_chunks:
        if config.openrouter_api_key:
            llm = OpenRouterLLM(api_key=config.openrouter_api_key)
        else:
            logger.warning("enrich_chunks is set but no API key — skipping enrichment")

    result = asyncio.run(
        ingest(
            config=config,
            embedder=embedder,
            store=store,
            llm=llm,
            recreate_collection=not no_recreate,
            embedding_batch_size=batch_size,
            enrich_batch_size=enrich_batch_size or 10,
        )
    )

    print("\nIngest complete:", file=sys.stderr)
    print(f"   Chunks created: {result['chunks']}", file=sys.stderr)
    print(f"     Code:      {result['code_chunks']}", file=sys.stderr)
    print(f"     Markdown:  {result['markdown_chunks']}", file=sys.stderr)
    print(f"     Plaintext: {result['plaintext_chunks']}", file=sys.stderr)
    print(f"   Indexed in Qdrant: {result['indexed']}", file=sys.stderr)
    print(f"   Pointer index entries: {result['pointer_entries']}", file=sys.stderr)
