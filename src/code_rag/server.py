"""
MCP Server: exposes document and code RAG tools to Claude Code.

Tools:
  - search_documents: semantic vector search across all content types
  - search_code: code-specific search with parent-class context
  - lookup_index: exact concept lookup against the prebuilt pointer index
    (no embeddings involved, so every result is a real file location)

The server is built by :func:`create_server`, which binds the tool handlers to
the providers passed in. ``main`` constructs the real OpenRouter and Qdrant
providers and starts the STDIO transport.
"""

import logging
import sys

from mcp.server.fastmcp import FastMCP

from .config import ProjectConfig, load_config
from .pointer_index import load_index, search_index
from .protocols import EmbeddingProvider, VectorStore

logger = logging.getLogger("code_rag.server")


def _format_result(r: dict, index: int, max_chars: int = 2000) -> str:
    """Format a single search result for display."""
    text = r["text"] or ""
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... [truncated, {len(r['text'])} chars total]"

    header = f"### Result {index} (score: {r['score']:.4f})"
    meta = f"**File:** `{r['file_name']}` — Lines {r['start_line']}–{r['end_line']}"

    if r.get("language"):
        meta += f" ({r['language']})"
    if r.get("symbol_name"):
        meta += f"\n**Symbol:** `{r['symbol_name']}` ({r.get('symbol_type', '')})"
    if r.get("summary"):
        meta += f"\n**Summary:** {r['summary']}"
    if r.get("math_density") and r["math_density"] > 0:
        meta += f"\n**Math density:** {r['math_density']:.2f}"

    # Format text as a code block for code chunks
    text_block = f"```{r['language']}\n{text}\n```" if r.get("language") else text

    return f"{header}\n{meta}\n\n{text_block}\n"


def create_server(
    config: ProjectConfig,
    embedder: EmbeddingProvider,
    store: VectorStore,
) -> FastMCP:
    """Build a FastMCP server whose tools are bound to the given providers."""
    mcp = FastMCP(config.name, instructions=config.description)

    @mcp.tool()
    async def search_documents(
        query: str,
        top_k: int = 5,
        min_math_density: float | None = None,
    ) -> str:
        """
        Search all indexed content (code, markdown, plaintext) for passages
        relevant to a query.

        Embeds the query and performs vector similarity search against the
        indexed chunks. Returns the most relevant passages with exact file
        locations.

        Args:
            query: Natural language search query.
            top_k: Number of results to return (default: 5, max: 20).
            min_math_density: Optional minimum math density filter (0.0-1.0).
                              Use to prefer formula-heavy sections.
        """
        top_k = min(top_k, 20)
        try:
            query_vector = (await embedder.embed([query]))[0]
            results = store.search(
                config.collection,
                query_vector,
                top_k=top_k,
                min_math_density=min_math_density,
            )

            if not results:
                return "No matching documents found for the given query."

            formatted = [_format_result(r, i) for i, r in enumerate(results, 1)]
            return "\n---\n".join(formatted)

        except Exception as e:
            logger.error("Search failed: %s", e, exc_info=True)
            return f"Error performing search: {e}"

    @mcp.tool()
    async def search_code(
        query: str,
        language: str | None = None,
        top_k: int = 5,
    ) -> str:
        """
        Search code files for functions, classes, and methods relevant to a query.

        When a method-level chunk is found, its parent class context is also
        returned so you can see the full class structure.

        Args:
            query: Natural language search query (e.g., "animation interpolation",
                   "how does Scene.play work", "shader vertex transformation").
            language: Optional language filter (e.g., "python", "glsl", "typescript").
            top_k: Number of results to return (default: 5, max: 20).
        """
        top_k = min(top_k, 20)
        try:
            query_vector = (await embedder.embed([query]))[0]
            results = store.search(
                config.collection,
                query_vector,
                top_k=top_k,
                language=language,
            )

            if not results:
                return "No matching code found for the given query."

            # Track parent_ids already shown to avoid duplicate context
            seen_parents: set[str] = set()

            formatted = []
            for i, r in enumerate(results, 1):
                block = _format_result(r, i)

                # If this is a child chunk, fetch and append parent context
                parent_id = r.get("parent_id")
                if parent_id and parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent = store.fetch_parent(config.collection, parent_id)
                    if parent:
                        parent_name = parent.get("symbol_name", "class")
                        parent_lang = parent.get("language", "")
                        parent_text = parent.get("text", "")
                        if len(parent_text) > 3000:
                            parent_text = parent_text[:3000] + "\n... [truncated]"
                        block += (
                            f"\n<details><summary>Parent context: {parent_name}</summary>\n\n"
                            f"```{parent_lang}\n{parent_text}\n```\n</details>"
                        )

                formatted.append(block)

            return "\n---\n".join(formatted)

        except Exception as e:
            logger.error("Code search failed: %s", e, exc_info=True)
            return f"Error performing code search: {e}"

    @mcp.tool()
    async def lookup_index(concept: str) -> str:
        """
        Look up a concept in the prebuilt index for exact file locations.

        Exact lookup, not semantic search: the concept is matched against an
        index built at ingest time, and every result is a real file path and
        line range taken from that index. No embeddings or vector search are
        involved. Use this when you know the name of the thing you want.

        Useful for finding specific definitions, theorems, function names,
        class names, or domain-specific terms.

        Args:
            concept: The concept to look up (e.g. "Definition 2.1",
                     "Scene", "VMobject.add", or a domain-specific term).
        """
        index_path = config.pointer_index

        if not index_path.exists():
            return "Pointer index not found. Run the ingest pipeline first:\n  code-rag ingest"

        try:
            index = load_index(index_path)
            results = search_index(index, concept)

            if not results:
                return f"No entries found for concept: '{concept}'"

            formatted = []
            for r in results[:10]:
                formatted.append(
                    f"- **{r['concept']}** ({r['entry_type']})\n"
                    f"  File: `{r['file_name']}` — Lines {r['start_line']}–{r['end_line']}\n"
                    f"  Path: `{r['file_path']}`"
                )

            header = f"Found {len(results)} entries for '{concept}'"
            if len(results) > 10:
                header += " (showing top 10)"

            return f"{header}\n\n" + "\n".join(formatted)

        except Exception as e:
            logger.error("Index lookup failed: %s", e, exc_info=True)
            return f"Error looking up index: {e}"

    return mcp


def main() -> None:
    """Entry point for the MCP server — wires real providers and serves over STDIO."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    from .providers import OpenRouterEmbeddings, QdrantVectorStore

    config = load_config()
    embedder = OpenRouterEmbeddings(api_key=config.openrouter_api_key)
    store = QdrantVectorStore(config.qdrant_url)
    mcp = create_server(config, embedder, store)

    logger.info("Starting %s MCP server (STDIO transport)", config.name)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
