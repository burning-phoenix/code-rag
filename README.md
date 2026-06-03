# code-rag

A reusable MCP server for code and document RAG. Drop source files (Python, Rust,
JS/TS, Go, Java, C/C++, Ruby, Swift, GLSL, Lean) and markdown into a project and get
semantic search + exact concept lookups inside Claude Code — each project with its
own Qdrant vector database and MCP server. AST-aware chunking keeps results tight;
optional LLM enrichment notably improves code retrieval.

## Install

```bash
pip install -e /path/to/code-rag
```

Requires Python 3.11+, Docker (for Qdrant), and an OpenRouter API key.

## Quick start

```bash
cd ~/projects/my-project
code-rag init                              # scaffold config, docker-compose, .mcp.json, RAG_data/
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
# drop files into RAG_data/, then set "file_extensions" in rag-config.json
docker compose up -d                       # start Qdrant
code-rag ingest                            # chunk, embed, index  (add --enrich for LLM summaries)
# open Claude Code in the project — the MCP tools are auto-discovered
```

## MCP tools

| Tool | What it does |
|---|---|
| `search_documents(query, top_k, min_math_density)` | Semantic vector search across all content types. |
| `search_code(query, language, top_k)` | Code-aware search; a retrieved method also returns its parent-class context. |
| `lookup_index(concept)` | Zero-hallucination exact concept→file/line lookup via a static index (no embeddings). |

## Configuration

`code-rag init` writes `rag-config.json`. Common fields: `collection`,
`data_dir` (default `RAG_data`), `file_extensions` (e.g. `[".py", ".md", ".lean"]`),
`enrich_chunks`, `key_terms`, and the chunk-size knobs. Environment variables
`OPENROUTER_API_KEY`, `QDRANT_URL`, and `RAG_PROJECT_DIR` override config at load
time. `OPENROUTER_API_KEY` is required (embeddings + enrichment).

## Supported file types

Python and the tree-sitter languages (JS/TS, Go, Rust, Java, C/C++, Ruby, Swift,
GLSL) are chunked by AST; Lean by declaration; Markdown by section header;
everything else by paragraph. Adding a language is one small file — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Retrieval quality

code-rag ships a span-level evaluation harness (recall@k, NDCG@k, MRR, line-range
IoU) with a 100-query golden set. Headline finding: AST chunking localizes relevant
regions far more tightly than naive line windows, and **LLM enrichment is a clear
win for code retrieval** (≈ +22% recall@10 on Python). Full methodology and numbers:
[REPORT.md](REPORT.md).

## Documentation

Architecture, how to extend it, the dev workflow, CI, and the evaluation are in
[CONTRIBUTING.md](CONTRIBUTING.md).
