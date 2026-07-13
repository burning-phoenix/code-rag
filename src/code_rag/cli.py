"""
CLI entry point: code-rag init | ingest | serve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ProjectConfig
    from .eval.dataset import GoldenDataset
    from .eval.runner import RunConfig


def _slugify(name: str) -> str:
    """Convert a directory name to a safe slug."""
    return name.lower().replace(" ", "-").replace("_", "-")


def _collection_name(name: str) -> str:
    """Convert a directory name to a valid Qdrant collection name."""
    return name.lower().replace(" ", "_").replace("-", "_")


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize a new code-rag project in the target directory."""
    project_dir = Path(args.project_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    dir_name = project_dir.name
    name = args.name or f"{_slugify(dir_name)}-rag"
    collection = args.collection or _collection_name(dir_name)
    container_name = f"{_slugify(dir_name)}-qdrant"
    qdrant_url = f"http://{container_name}.orb.local:6333"

    # 1. Create rag-config.json
    config = {
        "name": name,
        "collection": collection,
        "data_dir": "RAG_data",
        "pointer_index": "pointer_index.json",
        "qdrant_url": qdrant_url,
        "description": f"A local RAG server for {dir_name}. "
        f"Use 'search_documents' for semantic search, "
        f"'search_code' for code-aware search with class context, or "
        f"'lookup_index' for exact concept-to-location lookups.",
        "key_terms": [],
        "file_extensions": [".md"],
        "enrich_chunks": False,
    }

    config_path = project_dir / "rag-config.json"
    if config_path.exists():
        print("  [skip] rag-config.json already exists", file=sys.stderr)
    else:
        config_path.write_text(json.dumps(config, indent=2))
        print("  [created] rag-config.json", file=sys.stderr)

    # 2. Create docker-compose.yml
    compose_path = project_dir / "docker-compose.yml"
    if compose_path.exists():
        print("  [skip] docker-compose.yml already exists", file=sys.stderr)
    else:
        # Write YAML manually for clean formatting
        compose_content = (
            f"services:\n"
            f"  qdrant:\n"
            f"    image: qdrant/qdrant:latest\n"
            f"    container_name: {container_name}\n"
            f"    volumes:\n"
            f"      - ./qdrant_storage:/qdrant/storage\n"
            f"    restart: unless-stopped\n"
        )
        compose_path.write_text(compose_content)
        print("  [created] docker-compose.yml", file=sys.stderr)

    # 3. Create/merge .mcp.json
    python_exe = sys.executable
    mcp_entry = {
        "type": "stdio",
        "command": python_exe,
        "args": ["-m", "code_rag.server"],
        "env": {
            "RAG_PROJECT_DIR": str(project_dir),
        },
    }

    mcp_path = project_dir / ".mcp.json"
    if mcp_path.exists():
        existing = json.loads(mcp_path.read_text())
        if name in existing.get("mcpServers", {}):
            print(f"  [skip] .mcp.json already has '{name}' entry", file=sys.stderr)
        else:
            existing.setdefault("mcpServers", {})[name] = mcp_entry
            mcp_path.write_text(json.dumps(existing, indent=4))
            print(f"  [updated] .mcp.json — added '{name}'", file=sys.stderr)
    else:
        mcp_data = {"mcpServers": {name: mcp_entry}}
        mcp_path.write_text(json.dumps(mcp_data, indent=4))
        print("  [created] .mcp.json", file=sys.stderr)

    # 4. Create RAG_data directory
    data_dir = project_dir / "RAG_data"
    data_dir.mkdir(exist_ok=True)
    print("  [created] RAG_data/", file=sys.stderr)

    # 5. Create/append .gitignore entries
    gitignore_entries = [
        "qdrant_storage/",
        "pointer_index.json",
        ".env",
    ]
    gitignore_path = project_dir / ".gitignore"
    if gitignore_path.exists():
        existing_content = gitignore_path.read_text()
        new_entries = [e for e in gitignore_entries if e not in existing_content]
        if new_entries:
            with open(gitignore_path, "a") as f:
                f.write("\n# code-rag\n")
                for entry in new_entries:
                    f.write(f"{entry}\n")
            print("  [updated] .gitignore", file=sys.stderr)
        else:
            print("  [skip] .gitignore already has entries", file=sys.stderr)
    else:
        with open(gitignore_path, "w") as f:
            f.write("# code-rag\n")
            for entry in gitignore_entries:
                f.write(f"{entry}\n")
        print("  [created] .gitignore", file=sys.stderr)

    # 6. Create .env template
    env_path = project_dir / ".env"
    if not env_path.exists():
        env_path.write_text("# code-rag environment variables\nOPENROUTER_API_KEY=\n")
        print("  [created] .env (add your OPENROUTER_API_KEY)", file=sys.stderr)

    # Print summary
    print(f"\nProject initialized at {project_dir}", file=sys.stderr)
    print("\nNext steps:", file=sys.stderr)
    print("  1. Add your OPENROUTER_API_KEY to .env", file=sys.stderr)
    print("  2. Drop files into RAG_data/ (code, markdown, or any supported type)", file=sys.stderr)
    print('  3. Set file_extensions in rag-config.json (e.g. [".py", ".md"])', file=sys.stderr)
    print("  4. Start Qdrant:  docker compose up -d", file=sys.stderr)
    print("  5. Run ingestion: code-rag ingest", file=sys.stderr)
    print("  6. (Optional) Add --enrich flag to generate LLM summaries per chunk", file=sys.stderr)
    print("  7. (Optional) Add domain terms to key_terms in rag-config.json", file=sys.stderr)
    print(f"\nQdrant container: {container_name}", file=sys.stderr)
    print(f"Qdrant URL:       {qdrant_url}", file=sys.stderr)
    print(f"Collection:       {collection}", file=sys.stderr)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Run the ingest pipeline."""
    from .ingest import run_ingest

    run_ingest(
        project_dir=args.project_dir,
        no_recreate=args.no_recreate,
        batch_size=args.batch_size,
        min_chunk_lines=args.min_chunk_lines,
        max_chunk_lines=args.max_chunk_lines,
        enrich=args.enrich if args.enrich else None,
        enrich_batch_size=args.enrich_batch_size,
    )


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    if args.project_dir:
        os.environ["RAG_PROJECT_DIR"] = str(Path(args.project_dir).resolve())
    from .server import main as server_main

    server_main()


def cmd_eval(args: argparse.Namespace) -> None:
    """Run the retrieval evaluation against a golden dataset.

    Diagnostic only — requires an API key (to embed queries) and a running
    Qdrant. Indexes the dataset's corpus into an isolated collection, runs each
    query, and prints a metric table computed at each requested top_k value.
    """
    import asyncio
    import logging

    from .chunkers import chunk_directory
    from .config import load_config
    from .eval.dataset import load_dataset
    from .eval.metrics import QueryResult, load_corpus_weights
    from .eval.report import CANARY_LABEL, build_markdown_report, chunk_geometry
    from .eval.runner import EvalRunner, RunConfig
    from .providers import OpenRouterEmbeddings, OpenRouterLLM, QdrantVectorStore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = load_config(args.project_dir)

    dataset_path = Path(args.dataset).resolve()
    dataset = load_dataset(dataset_path)

    if not dataset.queries:
        print(
            f"Golden dataset {dataset_path} has no queries yet — nothing to evaluate.",
            file=sys.stderr,
        )
        print(
            f"Author queries there and add a corpus under '{dataset.corpus_dir}'.", file=sys.stderr
        )
        return

    # Point the pipeline at the eval corpus and an isolated collection.
    corpus_dir = (dataset_path.parent / dataset.corpus_dir).resolve()
    config.data_dir = corpus_dir
    config.collection = f"{config.collection}_eval"
    config.pointer_index = corpus_dir.parent / "eval_pointer_index.json"

    if not config.openrouter_api_key:
        print(
            "ERROR: OPENROUTER_API_KEY is required for eval (it embeds queries).", file=sys.stderr
        )
        sys.exit(1)

    top_k_values = [int(x) for x in args.top_k.split(",")]

    embedder = OpenRouterEmbeddings(api_key=config.openrouter_api_key)
    store = QdrantVectorStore(config.qdrant_url)
    llm = OpenRouterLLM(api_key=config.openrouter_api_key)

    # --matrix runs the standard comparison set; otherwise a single config from
    # flags. The whole-file configuration is always included: it exists to test
    # the metrics themselves (see canary_check in eval/report.py).
    # With --enrichment-models, one enriched column is added per model, labeled
    # with the model that wrote the enrichment, so reports are self-documenting.
    # The shared configs (ast / line-based / whole-file) are computed once.
    enrich_models = (
        [m.strip() for m in args.enrichment_models.split(",") if m.strip()]
        if args.enrichment_models
        else [config.enrichment_model]
    )

    def enrich_tag(model: str) -> str:
        return f"+enrich[{model.split('/')[-1]}]"

    if args.matrix:
        runs = [
            RunConfig(label=f"ast{enrich_tag(m)}", enrich=True, chunker="ast", enrichment_model=m)
            for m in enrich_models
        ] + [
            RunConfig(label="ast", enrich=False, chunker="ast"),
            RunConfig(label="line-based", enrich=False, chunker="line-based"),
            RunConfig(label=CANARY_LABEL, enrich=False, chunker="whole-file"),
        ]
    else:
        model = enrich_models[0]
        label = args.chunker + ("" if args.no_enrich else enrich_tag(model))
        runs = [
            RunConfig(
                label=label,
                enrich=not args.no_enrich,
                chunker=args.chunker,
                enrichment_model=model,
            )
        ]

    runner = EvalRunner(config, embedder, store, llm=llm, enrich_batch_size=args.enrich_batch_size)

    # Query vectors are config-independent: embed all queries once, batched,
    # and reuse them for every configuration in the matrix.
    query_vectors = asyncio.run(runner.embed_queries(dataset))

    results_by_label: dict[str, list[QueryResult]] = {}
    geometry_by_label: dict[str, dict[str, float]] = {}
    geometry_by_chunker: dict[str, dict[str, float]] = {}
    for run in runs:
        results_by_label[run.label] = asyncio.run(
            runner.evaluate(dataset, run, max(top_k_values), query_vectors)
        )
        # Geometry depends only on the chunker (enrichment doesn't move boundaries),
        # so compute once per distinct chunker and reuse. No API — pure chunking.
        if run.chunker not in geometry_by_chunker:
            chunks = chunk_directory(config.data_dir, config, run.registry())
            geometry_by_chunker[run.chunker] = chunk_geometry(chunks)
        geometry_by_label[run.label] = geometry_by_chunker[run.chunker]

    setup_md = _eval_setup_md(config, dataset, corpus_dir, runs, top_k_values)
    weights = load_corpus_weights(corpus_dir)
    report = build_markdown_report(
        results_by_label, geometry_by_label, top_k_values, setup_md, weights
    )
    print(report)
    if args.report_md:
        Path(args.report_md).write_text(report, encoding="utf-8")
        print(f"\n[report written to {args.report_md}]", file=sys.stderr)


def _eval_setup_md(
    config: ProjectConfig,
    dataset: GoldenDataset,
    corpus_dir: Path,
    runs: list[RunConfig],
    top_k_values: list[int],
) -> str:
    """Build the run-specific 'Setup' prose for the evaluation report."""
    import os
    from collections import Counter

    from .providers.openrouter import EMBEDDING_MODEL

    # Corpus: files + total lines per content type.
    corpus_lines: Counter = Counter()
    corpus_files: Counter = Counter()
    exts = set(config.file_extensions)
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix in exts:
            corpus_files[p.suffix] += 1
            text = p.read_text(encoding="utf-8", errors="replace")
            corpus_lines[p.suffix] += len(text.splitlines())
    corpus_rows = "\n".join(
        f"| `{ext}` | {corpus_files[ext]} | {corpus_lines[ext]} |" for ext in sorted(corpus_files)
    )

    # Dataset distribution.
    by_tool = Counter(q.tool for q in dataset.queries)
    span_ext = Counter(
        os.path.splitext(s.file)[1] for q in dataset.queries for s in q.decisive_spans
    )
    tool_line = ", ".join(f"{t}: {n}" for t, n in sorted(by_tool.items()))
    span_line = ", ".join(f"`{e}`: {n}" for e, n in sorted(span_ext.items()))
    config_line = " · ".join(f"`{r.label}`" for r in runs)
    enrich_line = " · ".join(
        f"`{r.enrichment_model or config.enrichment_model}`" for r in runs if r.enrich
    )

    return f"""\
## Setup

**Corpus** ({corpus_dir.name}/): {sum(corpus_files.values())} files indexed.

| ext | files | lines |
|---|---|---|
{corpus_rows}

**Golden dataset**: {len(dataset.queries)} hand-written queries — by tool: \
{tool_line}. Decisive gold spans by content type: {span_line}.

**Configurations**: {config_line}. **Embedding model**: `{EMBEDDING_MODEL}`. \
**Enrichment model(s)**: {enrich_line or "none (no enrich run)"}. \
**Retrieval**: top-k evaluated at {", ".join(str(k) for k in top_k_values)}.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-rag",
        description="Reusable MCP server for document RAG with Qdrant",
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new code-rag project",
    )
    init_parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory (default: current directory)",
    )
    init_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="MCP server name (default: <dir-name>-rag)",
    )
    init_parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: <dir_name>)",
    )

    # ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run the ingest pipeline (chunk, embed, index)",
    )
    ingest_parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Project directory (default: current directory)",
    )
    ingest_parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Update the existing collection in place instead of rebuilding it "
        "(replaces chunks of changed files, removes chunks of deleted files)",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Embedding batch size (default: 10)",
    )
    ingest_parser.add_argument(
        "--min-chunk-lines",
        type=int,
        default=None,
        help="Min chunk lines; smaller chunks merge into next sibling (default: config or 20)",
    )
    ingest_parser.add_argument(
        "--max-chunk-lines",
        type=int,
        default=None,
        help="Max chunk lines; larger chunks split at paragraph breaks (default: config or 100)",
    )
    ingest_parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help="Run LLM enrichment on chunks (adds summaries and hypothetical questions)",
    )
    ingest_parser.add_argument(
        "--enrich-batch-size",
        type=int,
        default=None,
        help="Number of chunks per LLM enrichment call (default: 10)",
    )

    # serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the MCP server (for debugging; normally started by Claude Code)",
    )
    serve_parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Project directory (default: current directory)",
    )

    # eval
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run retrieval evaluation against a golden dataset (needs API key + Qdrant)",
    )
    eval_parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Project directory (default: current directory)",
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default="tests/eval/golden_dataset.json",
        help="Path to the golden dataset JSON (default: tests/eval/golden_dataset.json)",
    )
    eval_parser.add_argument(
        "--top-k",
        type=str,
        default="3,5,7,10",
        help="Comma-separated top_k values to sweep (default: 3,5,7,10)",
    )
    eval_parser.add_argument(
        "--no-enrich",
        action="store_true",
        default=False,
        help="Index the corpus without LLM enrichment, to compare against enriched runs",
    )
    eval_parser.add_argument(
        "--chunker",
        choices=["ast", "line-based", "whole-file"],
        default="ast",
        help="Chunking strategy: structure-aware (default), fixed line windows, or "
        "whole files (the metric self-check)",
    )
    eval_parser.add_argument(
        "--matrix",
        action="store_true",
        default=False,
        help="Run the standard comparison set (ast+enrich / ast / line-based / "
        "whole-file), changing one thing at a time, and compare",
    )
    eval_parser.add_argument(
        "--enrich-batch-size",
        type=int,
        default=10,
        help="Chunks per LLM enrichment call during eval indexing (default: 10)",
    )
    eval_parser.add_argument(
        "--enrichment-models",
        type=str,
        default=None,
        help="Comma-separated enrichment models to compare; --matrix adds one "
        "enrich column per model (default: the project config's enrichment_model)",
    )
    eval_parser.add_argument(
        "--report-md",
        type=str,
        default="REPORT.md",
        help="Write the full Markdown report to this path (default: REPORT.md at the cwd; "
        "pass an empty string to skip)",
    )

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()
        sys.exit(1)
