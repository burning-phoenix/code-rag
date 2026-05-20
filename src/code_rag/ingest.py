"""
Ingest pipeline: chunks all files (markdown, code, plaintext), optionally
enriches with LLM metadata, embeds them, and indexes in Qdrant.
Also generates the static pointer index.
"""

import asyncio
import logging
import sys
from pathlib import Path

from .chunk_router import chunk_directory
from .config import ProjectConfig
from .embeddings import embed_texts
from .enrichment import get_embedding_text
from .indexer import get_client, ensure_collection, upsert_chunks
from .pointer_index import save_index

logger = logging.getLogger(__name__)


async def ingest(
    config: ProjectConfig,
    recreate_collection: bool = True,
    embedding_batch_size: int = 10,
    enrich_batch_size: int = 5,
) -> dict:
    """
    Run the full ingest pipeline:
    1. Chunk all files (markdown, code, plaintext)
    2. Generate static pointer index
    3. (Optional) Enrich chunks with LLM summaries
    4. Embed all chunks via OpenRouter
    5. Index embeddings in Qdrant

    Returns summary dict with counts.
    """
    data_dir = config.data_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Step 1: Chunk all files
    logger.info(
        "Chunking files in %s (extensions: %s) ...",
        data_dir, ", ".join(config.file_extensions),
    )
    chunks = chunk_directory(data_dir, config)

    code_chunks = sum(1 for c in chunks if c.chunk_type == "code")
    md_chunks = sum(1 for c in chunks if c.chunk_type == "markdown")
    pt_chunks = sum(1 for c in chunks if c.chunk_type == "plaintext")
    logger.info(
        "Created %d chunks (%d code, %d markdown, %d plaintext)",
        len(chunks), code_chunks, md_chunks, pt_chunks,
    )

    if not chunks:
        logger.warning(
            "No chunks created. Check that %s contains files with extensions: %s",
            data_dir, ", ".join(config.file_extensions),
        )
        return {"chunks": 0, "indexed": 0, "pointer_entries": 0}

    # Step 2: Generate static pointer index
    logger.info("Building static pointer index → %s", config.pointer_index)
    pointer_count = save_index(
        data_dir,
        config.pointer_index,
        key_terms=config.key_terms,
        file_extensions=config.file_extensions,
    )
    logger.info("Indexed %d pointer entries", pointer_count)

    # Step 3: Optional enrichment
    if config.enrich_chunks:
        from .enrichment import enrich_chunks
        logger.info(
            "Enriching %d chunks with LLM summaries (model: %s) ...",
            len(chunks), config.enrichment_model,
        )
        chunks = await enrich_chunks(
            chunks,
            api_key=config.openrouter_api_key,
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
            embeddings = await embed_texts(texts, api_key=config.openrouter_api_key)
        except Exception as e:
            logger.error(
                "Embedding failed at batch %d-%d / %d after retries: %s",
                i, min(i + embedding_batch_size, len(chunks)), len(chunks), e,
            )
            raise
        all_embeddings.extend(embeddings)
        logger.info(
            "Embedded %d / %d chunks",
            min(i + embedding_batch_size, len(chunks)),
            len(chunks),
        )

    # Step 5: Index in Qdrant
    logger.info("Connecting to Qdrant at %s ...", config.qdrant_url)
    client = get_client(config.qdrant_url)
    ensure_collection(client, collection_name=config.collection, recreate=recreate_collection)

    chunks_with_embeddings = list(zip(chunks, all_embeddings))
    indexed = upsert_chunks(client, chunks_with_embeddings, collection_name=config.collection)
    logger.info("Indexed %d chunks in Qdrant", indexed)

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
):
    """CLI-callable ingest runner."""
    from .config import load_config

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

    result = asyncio.run(ingest(
        config=config,
        recreate_collection=not no_recreate,
        embedding_batch_size=batch_size,
        enrich_batch_size=enrich_batch_size or 5,
    ))

    print(f"\nIngest complete:", file=sys.stderr)
    print(f"   Chunks created: {result['chunks']}", file=sys.stderr)
    print(f"     Code:      {result['code_chunks']}", file=sys.stderr)
    print(f"     Markdown:  {result['markdown_chunks']}", file=sys.stderr)
    print(f"     Plaintext: {result['plaintext_chunks']}", file=sys.stderr)
    print(f"   Indexed in Qdrant: {result['indexed']}", file=sys.stderr)
    print(f"   Pointer index entries: {result['pointer_entries']}", file=sys.stderr)
