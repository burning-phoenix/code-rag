# code-rag

A reusable MCP server for code and document retrieval-augmented generation. Drop source files (Python, GLSL, Rust, JavaScript, Lean, and more) and markdown documents into any project and get semantic search + exact concept lookups via Claude Code. There is an also a recommended enrichment module included for better semantic retrieval, specifically for code.

Each project gets its own Qdrant vector database (Docker container), its own document collection, and its own MCP server instance.

## Install

```bash
pip install -e /path/to/code-rag
```

## Quick Start

```bash
# 1. Initialize in your project
cd ~/projects/my-project
code-rag init

# 2. Add your OpenRouter API key
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env

# 3. Drop files into RAG_data/ (any supported format)

# 4. Update file_extensions in rag-config.json for your languages
#    e.g. [".py", ".md", ".glsl", ".lean", ".rs"]

# 5. Start Qdrant
docker compose up -d

# 6. Ingest documents
code-rag ingest

# 7. (Optional) Enrich chunks with LLM summaries
code-rag ingest --enrich

# 8. Open Claude Code in the project — MCP tools are auto-discovered
```

## What Gets Created

Running `code-rag init` scaffolds:

```
my-project/
├── rag-config.json       # Project-specific RAG configuration
├── docker-compose.yml    # Qdrant container
├── .mcp.json             # Claude Code MCP server registration
├── .env                  # API keys (gitignored)
├── .gitignore            # Ignores qdrant_storage/, pointer_index.json, .env
└── RAG_data/             # Put your source code and documents here
```

## MCP Tools

Once ingested, Claude Code gets three tools:

### `search_documents(query, top_k, min_math_density)`

Semantic vector search across **all** content types (code, markdown, plaintext). Embeds the query via OpenRouter and searches the Qdrant vector index.

```
query: "gradient descent convergence"
top_k: 5
min_math_density: 0.3  # optional: prefer formula-heavy sections
```

### `search_code(query, language, top_k)`

Code-specific semantic search with parent-child context retrieval. When a method-level chunk is found, the parent class source is automatically included in the response.

```
query: "how does Scene.play animate frames"
language: "python"     # optional: filter by programming language
top_k: 5
```

### `lookup_index(concept)`

Zero-hallucination exact lookup. Maps concepts directly to file paths and line numbers via a static pointer index — no embeddings involved.

```
concept: "Definition 2.1"
concept: "Scene"
concept: "VMobject.add"
concept: "backpropagation"  # if listed in key_terms
```

---

## Architecture

### Complete Data Flow

```
RAG_data/                        code-rag ingest
  ├── *.py, *.glsl, *.rs, ... ──→ Code Chunker (tree-sitter AST)
  ├── *.md                    ──→ Markdown Chunker (section headers)
  ├── *.lean                  ──→ Lean Chunker (regex declarations)
  └── *.* (fallback)          ──→ Plaintext Chunker (paragraphs)
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                       Pointer Index    ┌──(optional)──┐
                     (static JSON)      │   Enrichment  │
                                        │  (LLM batch)  │
                                        └──────┬────────┘
                                               ▼
                                    Embedding (OpenRouter)
                                         Qwen3-Embedding-8B
                                               │
                                               ▼
                                     Qdrant Vector Index
                                               │
                                ┌──────────────┼──────────────┐
                                ▼              ▼              ▼
                          search_documents  search_code  lookup_index
                             (MCP Tool)     (MCP Tool)   (MCP Tool)
```

### 1. Configuration System (`src/code_rag/config.py`)

`ProjectConfig` is the central dataclass that drives all behaviour. Resolution order for finding the project directory:

1. Explicit `project_dir` argument
2. `RAG_PROJECT_DIR` environment variable
3. Current working directory

The config is cached after first load (via `_cached_config`) to avoid repeated file I/O.

**Configuration fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `name` | `str` | `"code-rag"` | MCP server name (visible to Claude Code) |
| `collection` | `str` | `"documents"` | Qdrant collection name |
| `data_dir` | `Path` | `RAG_data/` | Directory containing files to index |
| `pointer_index` | `Path` | `pointer_index.json` | Static lookup index output path |
| `qdrant_url` | `str` | auto-generated | Qdrant HTTP endpoint |
| `description` | `str` | — | Instructions shown to Claude about this server |
| `key_terms` | `list[str]` | `[]` | Domain-specific terms for pointer index |
| `file_extensions` | `list[str]` | `[".md"]` | File extensions to process |
| `min_chunk_lines` | `int` | `20` | Min chunk size; smaller chunks merge into neighbours |
| `max_chunk_lines` | `int` | `100` | Max chunk size; larger chunks split at paragraph breaks |
| `code_chunk_min_lines` | `int` | `5` | Min code chunk size (runs after AST chunking) |
| `code_chunk_max_lines` | `int` | `150` | Code chunk size threshold for class splitting |
| `enrich_chunks` | `bool` | `False` | Whether to run LLM enrichment during ingest |
| `enrichment_model` | `str` | `"meta-llama/llama-4-scout"` | Model for chunk enrichment |

Environment variables (`QDRANT_URL`, `OPENROUTER_API_KEY`) override the corresponding config fields at load time.

### 2. Unified Chunk Data Model (`src/code_rag/chunker.py`)

All content types share a single `Chunk` dataclass:

| Field | Type | Purpose |
|-------|------|---------|
| `text` | `str` | The chunk's raw content |
| `file_name` | `str` | Source file name |
| `section_title` | `str` | Markdown header, function/class name, or file name |
| `start_line` | `int` | 1-indexed, inclusive |
| `end_line` | `int` | 1-indexed, inclusive |
| `math_density` | `float` | Ratio of math-containing lines (for markdown) |
| `chunk_id` | `str\|None` | Stable UUID assigned at creation time |
| `chunk_type` | `str` | `"markdown"`, `"code"`, or `"plaintext"` |
| `language` | `str\|None` | Programming language (e.g. `"python"`, `"glsl"`) |
| `symbol_name` | `str\|None` | Function/class/method name |
| `symbol_type` | `str\|None` | `"function"`, `"class"`, `"method"`, or `"module_preamble"` |
| `parent_id` | `str\|None` | `chunk_id` of parent class (links method → class) |
| `summary` | `str\|None` | LLM-generated summary (populated by enrichment) |
| `hypothetical_questions` | `list[str]\|None` | LLM-generated questions (populated by enrichment) |

### 3. Chunking Pipeline

#### 3a. Chunk Router (`src/code_rag/chunk_router.py`)

Dispatches each file to the appropriate chunker based on file extension:

| Extension | Chunker | Method |
|-----------|---------|--------|
| `.md` | Markdown | Section-header splitting |
| `.lean` | Lean | Regex-based declaration splitting |
| `.py`, `.js`, `.ts`, `.glsl`, `.rs`, `.go`, `.java`, `.c`, `.h`, `.cpp`, `.rb`, `.swift` | AST Code | tree-sitter AST-aware chunking |
| Everything else | Plaintext | Blank-line paragraph splitting |

#### 3b. Markdown Chunker (`src/code_rag/chunker.py`)

A **three-pass** pipeline:

1. **Header split**: splits at any `#` through `######` header line. Content before the first header gets the file name as its `section_title`.
2. **Merge pass** (`_merge_small_chunks`): chunks smaller than `min_chunk_lines` are merged forward into the next sibling. The last chunk (if small) is merged backward into the previous. The earlier section title is preserved.
3. **Split pass** (`_split_large_chunks`): chunks larger than `max_chunk_lines` are split at paragraph boundaries (blank lines). The splitter avoids breaking inside display math blocks (`$$...$$`). Continuation sub-chunks get a `## {title} (continued)` prefix so embeddings retain section context.

**Math density** (`_compute_math_density`): ratio of lines containing mathematical notation. Tracks display math blocks (`$$`), inline math (`$...$`), and common LaTeX commands (`\frac`, `\sum`, `\mathbb`, `^{`, `_{}`, etc.).

#### 3c. AST-Aware Code Chunker (`src/code_rag/code_chunker.py`)

Uses **tree-sitter** parsers for structural chunking. Supports 10 languages, each configured via `LANGUAGE_CONFIG` with node type mappings:

| Language | Extensions | Function nodes | Class nodes | Method nodes |
|----------|-----------|----------------|-------------|--------------|
| Python | `.py` | `function_definition` | `class_definition` | `function_definition` |
| JavaScript | `.js`, `.jsx`, `.mjs` | `function_declaration` | `class_declaration` | `method_definition` |
| TypeScript | `.ts`, `.tsx` | `function_declaration` | `class_declaration` | `method_definition` |
| GLSL | `.glsl`, `.vert`, `.frag`, `.comp` | `function_definition` | — | — |
| Rust | `.rs` | `function_item` | `struct_item`, `enum_item`, `impl_item` | `function_item` |
| Go | `.go` | `function_declaration`, `method_declaration` | `type_declaration` | — |
| Java | `.java` | — | `class_declaration`, `interface_declaration`, `enum_declaration` | `method_declaration`, `constructor_declaration` |
| C | `.c`, `.h` | `function_definition` | `struct_specifier` | — |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp` | `function_definition` | `class_specifier`, `struct_specifier` | `function_definition` |
| Ruby | `.rb` | `method` | `class`, `module` | `method` |
| Swift | `.swift` | `function_declaration` | `class_declaration`, `struct_declaration`, `protocol_declaration` | `function_declaration` |

Parsers are cached per language to avoid repeated initialisation.

**Chunking strategy:**

- **Module preamble**: all top-level nodes before the first function/class (imports, constants, comments) become a single chunk with `symbol_type="module_preamble"`.
- **Top-level functions**: each becomes its own chunk with `symbol_type="function"`.
- **Small classes** (≤ `code_chunk_max_lines`): single chunk for the entire class, `symbol_type="class"`.
- **Large classes** (> `code_chunk_max_lines`): a **parent chunk** containing the full class text, plus **child chunks** for each method. Child chunks are linked to the parent via `parent_id` — this enables the `search_code` tool to include full class context when a method chunk is retrieved.

A final `_merge_tiny_chunks` pass merges chunks below `code_chunk_min_lines` (default 5) into adjacent siblings of the same language.

#### 3d. Lean Chunker (`src/code_rag/lean_chunker.py`)

A lightweight regex-based chunker for the Lean proof assistant. Avoids introducing a tree-sitter Lean dependency. Handles:

- **Comment stripping**: tracks nested block comment depth (`/- ... -/`) and line comments (`--`)
- **Declaration detection**: regex matches top-level declarations (`def`, `theorem`, `lemma`, `structure`, `class`, `inductive`, `abbrev`, `instance`, `axiom`, `example`) including common modifiers (`private`, `protected`, `noncomputable`, `unsafe`, `partial`, `scoped`, `local`)
- **Semantic type mapping**: `theorem`/`lemma` → `"theorem"`/`"lemma"`; `structure`/`class`/`inductive` → `"class"`; `def`/`abbrev`/`axiom` → `"function"`
- **Module preamble**: content before the first declaration gets a preamble chunk

#### 3e. Plaintext Chunker (`src/code_rag/plaintext_chunker.py`)

Fallback for unsupported file types. Splits at blank-line paragraph boundaries with an accumulating merge strategy: paragraphs are combined until the accumulator would exceed `max_chunk_lines`, at which point it's flushed and a new accumulation begins. A final pass merges any remaining small chunks below `min_chunk_lines`.

### 4. Embedding System (`src/code_rag/embeddings.py`)

Calls OpenRouter's OpenAI-compatible embeddings API using **Qwen3-Embedding-8B** (4096-dimensional vectors).

- **Retry logic**: exponential backoff (2^x seconds) with up to 2 retries on transient failures
- **Rate limiting**: 429 responses are retried with backoff
- **Dynamic timeout**: `30 + 10 × batch_size` seconds, scaling with batch size
- Results are sorted by `index` to guarantee correct ordering

### 5. Enrichment System (`src/code_rag/enrichment.py`)

Optional LLM-based metadata generation that adds `summary` and `hypothetical_questions` to each chunk. Uses a **soul + constitution prompting** paradigm for reliable structured output:

- **Soul**: `"CodeSage"` identity primes the model for technical summarisation
- **Constitution**: 6 strict rules mandating JSON-only output with exact schema (`id`, `summary`, `hypothetical_questions` array, one entry per input chunk)
- **`response_format: json_object`**: enforced server-side

**Batching**: N chunks are sent in a single LLM call (default 5), dividing API costs by N. Each chunk prompt includes chunk type, file name, language, and line range. Chunk text is truncated to 2000 characters.

**Two-stage parsing** (`_parse_batch_response`):

1. Direct `json.loads()` + Pydantic validation (`BatchEnrichmentResponse`)
2. Regex extraction to handle cases where the model wraps JSON in markdown fences or adds preamble

**Robustness features:**

- ID-based sorting of batch results with positional fallback if IDs are garbled
- Pydantic field validators reject empty summaries
- **Retry with error feedback**: on parse failure, the previous (malformed) assistant response is included in the conversation context along with the specific parse error, and the model is asked to correct itself
- **Batch → individual fallback**: if an entire batch fails all retries, each chunk is retried individually
- Chunks shorter than 3 lines are skipped (not enough context for meaningful enrichment)
- Concurrency limited by `asyncio.Semaphore` (default 3 concurrent batch calls)

**`get_embedding_text`**: after enrichment, the embedding text is the concatenation of the original chunk text + summary + hypothetical questions. This means the vector captures both the raw content and its semantic metadata, improving retrieval quality.

### 6. Qdrant Indexer (`src/code_rag/indexer.py`)

Manages the Qdrant vector collection:

- **Collection creation**: Cosine distance, 4096-dimensional vectors
- **Deterministic point IDs**: MD5 hash of `file_name:chunk_type:start_line-end_line` converted to UUID format — this ensures idempotent re-ingestion (upserting the same chunk again uses the same ID)
- **Full payload**: all chunk fields are stored as Qdrant payload for retrieval
- **Search**: uses `query_points` (Qdrant's own NN search). Supports optional filters:
  - `min_math_density`: range filter for math-heavy content
  - `language`: exact match filter for programming language
- **`fetch_parent`**: scrolls the collection by `chunk_id` to retrieve a parent class chunk (used by `search_code` for context expansion)

### 7. Pointer Index (`src/code_rag/pointer_index.py`)

A static JSON index that maps concepts to exact file locations — no embeddings, zero hallucination risk.

**`IndexEntry` dataclass:**

| Field | Type | Purpose |
|-------|------|---------|
| `concept` | `str` | The lookup key |
| `file_name` | `str` | Source file name |
| `file_path` | `str` | Absolute path |
| `start_line` | `int` | 1-indexed |
| `end_line` | `int` | 1-indexed |
| `entry_type` | `str` | `"section"`, `"definition"`, `"theorem"`, `"class"`, `"function"`, `"method"`, `"term"`, etc. |

**Markdown scanning** (`_scan_file`):

- Section headers (`#` through `######`) become `"section"` entries
- Formal patterns: `**Definition X.Y**`, `**Theorem X.Y**`, `**Proposition X.Y**`, `**Corollary X.Y**`, `**Lemma X.Y**`, `**Remark X.Y**`, `**Example X.Y**` — each maps to its type
- Key terms: matches configured `key_terms` against file content (case-insensitive substring), skips duplicates

**Code scanning** (`_scan_code_file`):

- Uses the same tree-sitter infrastructure as the code chunker
- Extracts top-level functions → `"function"` entries
- Extracts classes/structs → `"class"` entries
- Extracts methods within classes → `"method"` entries (named `ClassName.methodName`)

**Lean scanning** (`_scan_lean_file`):

- Same regex declaration detection as the Lean chunker
- Maps declaration kinds to entry types: `def`/`abbrev` → `"definition"`, `theorem`/`lemma` → `"theorem"`/`"lemma"`, `structure`/`class`/`inductive` → `"class"`

**Search** (`search_index`): case-insensitive substring matching, sorted by exact-match priority then by entry-type priority (definitions first, terms last).

### 8. Ingest Pipeline (`src/code_rag/ingest.py`)

Orchestrates the full pipeline in sequence:

1. **Chunk** all files via the chunk router
2. **Build pointer index** from chunked files (static JSON)
3. **Enrich** (optional): batch-LLM summaries and questions
4. **Embed** all chunks via OpenRouter (using enriched text when available)
5. **Index** embeddings in Qdrant with deterministic IDs

CLI flags (`--min-chunk-lines`, `--max-chunk-lines`, `--enrich`) override config values at runtime.

### 9. MCP Server (`src/code_rag/server.py`)

Built on FastMCP, communicating over stdio. Three tools exposed to Claude Code:

**`search_documents`** — Generic semantic search:
1. Embeds the user query via OpenRouter
2. Queries Qdrant with vector similarity
3. Formats results with score, file location, language, symbol info, summary, math density
4. Code chunks are wrapped in markdown code fences

**`search_code`** — Code-aware semantic search:
1. Same embedding + search flow as `search_documents`, with optional language filter
2. Result formatting includes parent class context: when a method chunk with a `parent_id` is found, the parent class source is fetched from Qdrant and appended in a collapsible `<details>` block
3. Parent deduplication: if multiple results share the same parent class, it's only shown once

**`lookup_index`** — Zero-hallucination static lookup:
1. Loads the `pointer_index.json` file
2. Case-insensitive substring search
3. Returns matched entries with concept, type, file, and line range

All results are limited to `top_k=20` and text is truncated to 2000 characters.

### 10. CLI (`src/code_rag/cli.py`)

Three subcommands:

**`code-rag init`** — Scaffolds a new project:
- `rag-config.json` with auto-generated Qdrant URL (OrbStack DNS)
- `docker-compose.yml` with Qdrant service
- `.mcp.json` registering the MCP server for Claude Code
- `RAG_data/` directory
- `.gitignore` entries for `qdrant_storage/`, `pointer_index.json`, `.env`
- `.env` template for `OPENROUTER_API_KEY`
- Smart merging: won't overwrite existing config files or duplicate `.mcp.json` entries

**`code-rag ingest`** — Runs the full pipeline:
- `--enrich` to enable LLM chunk enrichment
- `--enrich-batch-size` for enrichment batch control (default 5)
- `--batch-size` for embedding batches (default 10)
- `--no-recreate` to append to an existing collection
- `--min-chunk-lines` / `--max-chunk-lines` to override config chunk sizes

**`code-rag serve`** — Starts the MCP server manually (for debugging; normally Claude Code starts it automatically)

## Configuration Reference

`rag-config.json` full schema:

```json
{
    "name": "my-project-rag",
    "collection": "my_project",
    "data_dir": "RAG_data",
    "pointer_index": "pointer_index.json",
    "qdrant_url": "http://my-project-qdrant.orb.local:6333",
    "description": "A local RAG server for my-project.",
    "key_terms": ["gradient descent", "loss function", "backpropagation"],
    "file_extensions": [".py", ".md", ".glsl"],
    "min_chunk_lines": 20,
    "max_chunk_lines": 100,
    "code_chunk_min_lines": 5,
    "code_chunk_max_lines": 150,
    "enrich_chunks": false,
    "enrichment_model": "meta-llama/llama-4-scout"
}
```

### Environment Variables

Set in `.env` or export directly:

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | **Required.** API key for embeddings (Qwen3-Embedding-8B) and enrichment LLM calls |
| `QDRANT_URL` | Override the Qdrant URL from config |
| `RAG_PROJECT_DIR` | Override the project directory (used internally by MCP server) |

## Supported Languages

| Language | Extensions | Chunking method |
|----------|-----------|-----------------|
| Python | `.py` | tree-sitter AST |
| JavaScript | `.js`, `.jsx`, `.mjs` | tree-sitter AST |
| TypeScript | `.ts`, `.tsx` | tree-sitter AST |
| GLSL | `.glsl`, `.vert`, `.frag`, `.comp` | tree-sitter AST |
| Rust | `.rs` | tree-sitter AST |
| Go | `.go` | tree-sitter AST |
| Java | `.java` | tree-sitter AST |
| C | `.c`, `.h` | tree-sitter AST |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp` | tree-sitter AST |
| Ruby | `.rb` | tree-sitter AST |
| Swift | `.swift` | tree-sitter AST |
| Lean | `.lean` | regex (declarations) |
| Markdown | `.md` | section headers |
| Plaintext | all others | paragraph boundaries |

## Multiple Projects

Each project is fully isolated:
- Own Qdrant Docker container (unique container name)
- Own collection, pointer index, and data directory
- Own `.mcp.json` entry

You can run multiple projects simultaneously.

## Requirements

- Python 3.11+
- Docker (for Qdrant)
- OpenRouter API key (for embeddings and optional enrichment)
