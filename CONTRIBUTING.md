# Contributing to code-rag

This document is for people extending code-rag. It covers the architecture, how to
add a file type or swap a provider, the dev workflow, CI, and the retrieval-quality
evaluation.

## Design philosophy

Read [SOLID Principles](https://realpython.com/solid-principles-python/) to build 
upon this project. Abstractions exist only where there was real friction 
(dispatch on file type; hard-coded external services).
Two rules keep the codebase extensible:

- **Dispatch is data, not code.** A `ChunkerRegistry` maps file extension → chunker.
- **Orchestrators depend on protocols, never concretes.** `ingest`, `server`, and
  `enrichment` are written against `EmbeddingProvider` / `VectorStore` /
  `LLMProvider` (`typing.Protocol`); the composition roots build the real
  OpenRouter/Qdrant implementations, and tests inject fakes.

## Architecture

```
src/code_rag/
├── cli.py            # argparse dispatcher: init | ingest | serve | eval
├── config.py         # ProjectConfig dataclass + load_config()
├── models.py         # Chunk, IndexEntry dataclasses + point_id() (home for chunk identity)
├── protocols.py      # EmbeddingProvider, VectorStore, LLMProvider (Protocols) + BaseChunker (ABC)
├── chunkers/         # one BaseChunker per file type + the registry
│   ├── __init__.py   #   ChunkerRegistry, get_registry(), chunk_directory()
│   ├── markdown.py   #   MarkdownChunker  (section headers)
│   ├── python.py     #   PythonChunker    (stdlib ast; .py — falls back to CodeChunker on SyntaxError)
│   ├── code.py       #   CodeChunker      (tree-sitter; the other 10 languages)
│   ├── lean.py       #   LeanChunker      (regex declarations + doc/attribute attachment)
│   └── plaintext.py  #   PlaintextChunker (registry fallback)
├── providers/        # concrete implementations of the protocols
│   ├── openrouter.py #   OpenRouterEmbeddings (Qwen3-Embedding-8B, 4096-dim), OpenRouterLLM
│   ├── qdrant.py     #   QdrantVectorStore
│   └── fake.py       #   FakeEmbeddings, InMemoryVectorStore, FakeLLM (test doubles)
├── enrichment.py     # enrich_chunks(chunks, llm, ...) — batched LLM summaries + questions
├── ingest.py         # ingest(config, embedder, store, llm=None, ...) + run_ingest() composition root
├── server.py         # _format_result() + create_server(config, embedder, store) factory + main()
├── pointer_index.py  # build/save/load/search the static concept→location index
└── eval/             # retrieval evaluation framework (diagnostic, not in CI)
```

### Pipeline

`code-rag ingest` runs: **chunk** every file (registry dispatch) → **build the
pointer index** (each chunker's `scan()`) → optional **LLM enrichment** → **embed**
(OpenRouter) → **index** in Qdrant. The MCP server then exposes `search_documents`,
`search_code`, and `lookup_index`.

### Key invariants

- **Chunk identity has one home: `models.point_id()`.** Both stores derive Qdrant
  point IDs through it, keyed on the corpus-relative `source_path` (stamped by
  `chunk_directory`) so same-named files in different folders never collide.
- **`ingest --no-recreate` is a true incremental sync** (`ingest._sync_incremental`):
  per-file delete-then-upsert clears chunks whose line ranges shifted on edit, and
  source paths no longer present are pruned. The sync *policy* lives in `ingest`; the
  store only exposes `delete_by_source` / `list_source_paths`.
- **`server.create_server()` is a factory** — importing `code_rag.server` is
  side-effect-free and never requires a config file.

### Chunkers at a glance

| Chunker | Extensions | Strategy |
|---|---|---|
| `PythonChunker` | `.py` | stdlib `ast`: per-method chunks (substantial methods), decorators, large-function splitting; tree-sitter fallback on `SyntaxError` |
| `CodeChunker` | `.js/.ts/.glsl/.rs/.go/.java/.c/.cpp/.rb/.swift` | tree-sitter AST; classes split into parent + method children when large |
| `LeanChunker` | `.lean` | regex declarations; leading doc-comments/attributes attach to their declaration |
| `MarkdownChunker` | `.md` | section headers, with merge/split passes and math-density |
| `PlaintextChunker` | fallback | blank-line paragraphs |

## Extending

**Add a file type / language** (Open/Closed — no edits to existing dispatch):

1. Create `chunkers/<x>.py` implementing `BaseChunker` (`supported_extensions()`,
   `chunk()`, and optionally `scan()` for the pointer index).
2. Add one line to `get_registry()` in `chunkers/__init__.py`:
   `registry.register(MyChunker())`. (Registering later wins for shared extensions —
   that is how `PythonChunker` claims `.py` from `CodeChunker`.)

`source_path`, deterministic point IDs, embedding, and indexing are handled for you.

**Swap a provider** (embeddings / vector store / LLM): implement the relevant
`Protocol` in `protocols.py` and inject it at a composition root (`run_ingest`,
`server.main`, or `EvalRunner`). No orchestrator code changes.

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/code_rag/
pytest -m "not live"        # unit suite — no network, no Docker
pytest -m live              # needs OPENROUTER_API_KEY (+ Qdrant for some)
pytest tests/test_snapshots.py --update-snapshots   # regenerate chunker snapshots
```

**Testing conventions:** unit tests use the fakes in `providers/fake.py` and small
fixtures in `tests/fixtures/`; external-dependency tests are marked
`@pytest.mark.live` and excluded from CI. Snapshot tests pin chunker output (random
`chunk_id`/`parent_id` UUIDs are normalised to order tokens).

**Coding conventions:** Python 3.11+, **type hints on every function** (CI runs
mypy with `disallow_untyped_defs = true`), Google-style docstrings. Blind
`except Exception` only at true boundaries, tagged `# noqa: BLE001` with a reason.
`# type: ignore` only for untyped tree-sitter nodes (in `chunkers/code.py`), with
an inline reason.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`, across a
Python **3.11 / 3.12** matrix:

1. `pip install -e ".[dev]"`
2. `ruff check src/ tests/`
3. `ruff format --check src/ tests/`
4. `mypy src/code_rag/`
5. `pytest -m "not live"`

All four gates must be green. CI is **hermetic** — no network and no Docker; the
`live` tests and the evaluation are intentionally excluded (they need an API key and
a running Qdrant). The eval corpus under `tests/eval/corpus/` is third-party sample
data and is excluded from ruff via `extend-exclude` in `pyproject.toml`.

## Evaluation

`src/code_rag/eval/` is a retrieval-quality harness with span-level graded relevance
(decisive / supportive) and metrics: recall@k, NDCG@k (span-coverage, capped ≤1),
MRR@k, hit-rate, and **line-range IoU@k** (localization precision). It is diagnostic,
never a CI gate.

```bash
code-rag eval --project-dir tests/eval --dataset tests/eval/golden_dataset.json --matrix
```

`--matrix` runs the standard ablation (AST+enrich / AST / line-based) and emits a
full Markdown report (metric definitions, sample sizes, per-content-type tables) to
`REPORT.md` at the repo root by default (override with `--report-md PATH`). See
[`REPORT.md`](REPORT.md) for the latest results. The golden dataset
(`tests/eval/golden_dataset.json`) is hand-written — do **not** auto-generate
queries with an LLM, as that biases the enrichment ablation.

## Roadmap — higher quality & more extensibility

Good first contributions, grouped by theme. None of these are blockers; they make
the project more robust and easier to build on.

### Extensibility

- **Provider registry.** Chunkers are fully drop-in (new file + one `register()`
  line), but providers are not: the composition roots (`ingest.run_ingest`,
  `server.main`, `cmd_eval`) hard-code `OpenRouterEmbeddings` / `QdrantVectorStore`
  / `OpenRouterLLM`. Add a provider factory/registry mirroring `ChunkerRegistry`,
  selected from `rag-config.json` (e.g. `"vector_store": "qdrant" | "pinecone"`,
  `"embeddings": "openrouter" | …`). The Protocols already exist — this is just the
  selection layer, and it makes a new backend (Pinecone, pgvector, a different
  embedding API) config-driven instead of a wiring edit.
- **Pluggable enrichment strategy.** The soul+constitution prompt and parser live
  in `enrichment.py`. Extract them behind a small interface so alternative
  enrichment schemes are swappable like providers are.
- **Tool registry.** Adding an MCP tool means editing `create_server()`; a small
  registry would make tools additive too (lower priority).

### Type safety & lint

- **mypy Tier 2.** Enable `disallow_any_generics` and parameterize the remaining
  ~39 bare `dict`/`list`/`tuple` generics (most become `dict[str, Any]`; a few
  unlock real checking), then move toward full `--strict` incrementally.
- **ruff polish.** The bug-catching families (`F`, `B`) are already on. Optionally
  add `RUF` (mutable-default-in-dataclass, stale `# noqa`) and the auto-fixable
  `C4` / `RET` / `PERF` / `FURB` for cleanups — these are quality, not bug fixes.

### Test coverage

CI is hermetic (fakes only), so the real I/O boundaries are unit-untested:

- **Provider boundaries.** `providers/openrouter.py` (HTTP retry / 429 backoff) and
  `providers/qdrant.py` (filter / scroll / delete) are exercised only by the `live`
  tests and manual eval runs. Add mocked-HTTP tests (e.g. `respx`) for retry/backoff
  and a `live`-gated integration job for Qdrant.
- **Orchestration.** `EvalRunner`, the `create_server()` tool closures, and the
  `cli` command functions have no unit tests (the tool closures can use the fakes).
- **Coverage metric.** Add `pytest-cov` and a threshold so "N tests" becomes a
  coverage number.

### Evaluation

- **Confidence intervals.** Report bootstrap CIs over queries so small per-type
  deltas come with error bars instead of reading as fact.
- **Bigger golden set per content type** (~50 each) for statistical power on the
  per-type claims (see the sample-size note in `REPORT.md`).
- **Content-type-aware enrichment.** The eval shows enrichment clearly helps code
  (≈ +22% recall@10 on `.py`) but is neutral-to-negative on prose. Make
  `enrich_chunks` skip markdown (or take a per-extension policy from config) — an
  evidence-backed default that also saves LLM cost.
- **Token-aware `lookup_index`.** `search_index` does full-string substring
  matching, so natural-language queries pinned to `lookup_index` under-retrieve.
  Splitting the query into terms and ranking concepts by term overlap would let it
  answer NL queries (no embeddings needed).

### Known limitations (acceptable today, worth fixing)

- **Giant top-level functions** in the tree-sitter languages (not Python — that
  already splits) become one oversized chunk, and there is no pre-embed truncation,
  so a very large chunk can exceed the embedding model's input limit mid-ingest.
- `rglob` follows **symlinks** (possible recursion); non-UTF-8 / binary files,
  unbalanced `$$` in markdown (skews `math_density`), and an unterminated Lean `/-`
  block comment (swallows following declarations) are all edge cases to harden.
