# code-rag

An MCP server that gives an AI coding agent (such as Claude Code) search tools
over your own files. Point it at a directory of source code and documents, run
one ingest command, and the agent can search that content semantically, search
code with class context, and look up exact definitions by name. Each project
gets its own vector database and its own server instance, so projects never
mix.

Code is split along its real structure — functions, classes, methods — instead
of at arbitrary line counts. This does not change how often the answer is
found, but it means results contain about 1.6× more relevant text per token
read, and answers arrive as complete units instead of fragments (measured
results below). An optional enrichment step writes an LLM summary for each
chunk, which measurably improves code search.

Supported languages: Python, Rust, JavaScript/TypeScript, Go, Java, C/C++,
Ruby, Swift, GLSL, and Lean, plus Markdown and plain text.

## Requirements

- Python 3.11 or newer
- Docker — runs Qdrant, the vector database each project stores its search
  index in
- An OpenRouter API key. **The key is required even if you never use
  enrichment**: turning text into search vectors (both at ingest time and for
  every query) also goes through OpenRouter.

## Install

```bash
pip install -e /path/to/code-rag
```

## Setting up a project

```bash
cd ~/projects/my-project
code-rag init
```

`init` creates `rag-config.json`, `docker-compose.yml`, `.mcp.json`, an empty
`RAG_data/` directory, a `.env` template, and `.gitignore` entries. Then:

1. **Add your API key** to `.env`: `OPENROUTER_API_KEY=sk-or-...`

2. **Copy the files you want searchable into `RAG_data/`.** Subdirectories are
   fine; the whole tree is scanned.

3. **List your file types in `rag-config.json`.** Set `"file_extensions"` to
   match what you added, for example `[".py", ".md", ".lean"]`. The default is
   `[".md"]` only — **a file whose extension is not in this list is skipped
   without any error message**. If ingest reports zero chunks, this list is
   the first thing to check.

4. **Check `"qdrant_url"`.** `init` writes a URL in the form
   `http://<project>-qdrant.orb.local:6333`, which works if your Docker runs
   under OrbStack (macOS). On Docker Desktop or plain Docker on Linux, that
   hostname does not resolve. Fix it in two steps: change `"qdrant_url"` to
   `"http://localhost:6333"`, and add a port mapping to the generated
   `docker-compose.yml` (it does not publish a port by default):

   ```yaml
   services:
     qdrant:
       image: qdrant/qdrant:latest
       container_name: my-project-qdrant
       ports:
         - "6333:6333"
       volumes:
         - ./qdrant_storage:/qdrant/storage
       restart: unless-stopped
   ```

5. **Start Qdrant**: `docker compose up -d`. Qdrant must be running before
   every ingest and whenever the server is used.

6. **Build the index**: `code-rag ingest`. Add `--enrich` to also generate an
   LLM summary per chunk — slower and it costs API credits, but it improves
   code search (numbers below).

   All `ingest` flags:

   | Flag | Effect |
   |---|---|
   | `--enrich` | Generate an LLM summary and hypothetical questions per chunk before embedding. Off by default; can also be turned on permanently via `"enrich_chunks": true` in `rag-config.json`. |
   | `--enrich-batch-size N` | Chunks summarised per LLM call (default 10). Larger batches mean fewer API calls; if a batch fails validation twice, its chunks are retried one at a time. |
   | `--batch-size N` | Chunks embedded per API call (default 10). |
   | `--no-recreate` | Update the existing collection in place instead of dropping and rebuilding it: chunks of changed files are replaced, chunks of deleted files are removed. Faster on large projects. |
   | `--min-chunk-lines N` | Markdown chunks smaller than N lines are merged into the next section (default: config value, or 20). |
   | `--max-chunk-lines N` | Markdown chunks larger than N lines are split at paragraph breaks (default: config value, or 100). |
   | `--project-dir PATH` | Ingest a project other than the current directory. |

7. **Open Claude Code in the project directory.** It reads `.mcp.json` and the
   three tools appear automatically. No extra registration step.

Two things worth knowing about the generated `.mcp.json`:

- It records the **absolute path of the Python interpreter** that ran
  `code-rag init`. If you later delete, move, or rebuild that virtual
  environment, the server silently stops starting — update the `"command"`
  field to the new interpreter path.
- The server reads the project location from the `RAG_PROJECT_DIR` variable in
  that file, so it works no matter which directory the agent launches it from.

## Keeping the index up to date

The index is a snapshot: after adding, editing, or deleting files in
`RAG_data/`, run `code-rag ingest` again. By default this rebuilds the whole
collection. `code-rag ingest --no-recreate` instead updates in place — it
replaces the chunks of files present in this run and removes the chunks of
files you deleted — which is faster on large projects.

## The three tools

| Tool | What it does |
|---|---|
| `search_documents(query, top_k, min_math_density)` | Semantic search across everything indexed. `min_math_density` (0.0–1.0) restricts results to formula-heavy sections. |
| `search_code(query, language, top_k)` | Semantic search over code. When a result is a method, its parent class is attached so the agent sees where the method lives. `language` restricts results to one language (e.g. `"python"`). |
| `lookup_index(concept)` | Exact lookup, no embeddings involved. Matches the concept string against a prebuilt index of definitions, theorems, classes, functions, and section titles, and returns real file paths and line numbers. Use it when you know the name of the thing you want. |

`top_k` is capped at 20 for both search tools.

## Configuration

`code-rag init` writes `rag-config.json` with these fields:

| Field | Meaning | Default |
|---|---|---|
| `name` | Server name shown to the agent | `<directory>-rag` |
| `collection` | Qdrant collection the index lives in | derived from directory name |
| `data_dir` | Directory scanned at ingest | `RAG_data` |
| `file_extensions` | File types to index; everything else is skipped | `[".md"]` |
| `enrich_chunks` | Generate LLM summaries at ingest | `false` |
| `enrichment_model` | Model that writes the summaries | `meta-llama/llama-4-scout` |
| `key_terms` | Domain terms to add to the exact-lookup index | `[]` |
| `min_chunk_lines`, `max_chunk_lines` | Size bounds for markdown chunks | 20 / 100 |
| `code_chunk_min_lines`, `code_chunk_max_lines` | Size bounds for code chunks | 5 / 150 |

Three environment variables override the config at load time:
`OPENROUTER_API_KEY` (required), `QDRANT_URL`, and `RAG_PROJECT_DIR` (which
directory to treat as the project).

## How files are split

Python is parsed with the standard-library parser; the ten other programming
languages are parsed with tree-sitter. Each function, class, and substantial
method becomes its own chunk. Lean files are split at declarations (`def`,
`theorem`, `structure`, ...). Markdown is split at section headers. Any other
listed file type is split at blank lines. Adding a language is one small file —
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Measured search quality

The repository includes an evaluation: 100 hand-written queries over a 16-file
corpus (Python, Lean, and four machine-learning papers converted to markdown),
each query paired with hand-labelled answer locations. Six indexing
configurations are compared — structure-aware chunking with each of three
enrichment models, structure-aware chunking without enrichment, fixed 40-line
windows, and whole files — on three metrics, each computed at result-set sizes
k = 3, 5, 7, and 10: **coverage** (did the answer text arrive in the
results?), **concentration** (what fraction of the retrieved text was
answer?), and **wholeness** (did each answer arrive inside a single chunk?).

What the evaluation found:

- **Finding the answer does not separate the chunking strategies.** Fixed
  40-line windows match structure-aware chunking on coverage@10 (0.831 vs
  0.814). Judged on coverage alone, parsing code buys nothing.
- **Reading cost and answer integrity do separate them.** Structure-aware
  chunks deliver about 1.6× more answer per retrieved token at every k, and on
  Python every answer that arrived came inside a single chunk (wholeness
  1.000, against 0.774 for line windows — roughly a quarter of answers
  arriving in pieces).
- **Which model writes the enrichment matters, where search can be improved
  at all.** Comparing three models (llama-4-scout, gemma-4-31b,
  deepseek-v4-flash): gemma's summaries raised Python coverage at k=3 from
  0.617 to 0.763 (+24% relative), clearly ahead of the other two — and gemma
  is the cheapest of the three. On Lean, no model made enrichment a net
  positive; better summaries only shrank the loss (scout −0.063, gemma
  −0.024). English summaries do not fix search over formal mathematical
  notation.
- **Chunking controls cost and integrity; enrichment controls whether the
  answer is found.** Concentration and wholeness barely change with the
  enrichment model.
- **The metric self-check passed.** One configuration indexes whole files
  purely to test the metrics: it posts the highest coverage (0.971) and by far
  the lowest concentration (0.003), and is never ranked best overall — exactly
  the behaviour the metrics are supposed to enforce.

Full method and tables: [REPORT.md](REPORT.md).

## More documentation

Architecture, how to add a language or swap a backend, the development
workflow, CI, and how to run the evaluation yourself:
[CONTRIBUTING.md](CONTRIBUTING.md).