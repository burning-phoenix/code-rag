"""
CLI entry point: code-rag init | ingest | serve
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _slugify(name: str) -> str:
    """Convert a directory name to a safe slug."""
    return name.lower().replace(" ", "-").replace("_", "-")


def _collection_name(name: str) -> str:
    """Convert a directory name to a valid Qdrant collection name."""
    return name.lower().replace(" ", "_").replace("-", "_")


def cmd_init(args):
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
        print(f"  [skip] rag-config.json already exists", file=sys.stderr)
    else:
        config_path.write_text(json.dumps(config, indent=2))
        print(f"  [created] rag-config.json", file=sys.stderr)

    # 2. Create docker-compose.yml
    compose = {
        "services": {
            "qdrant": {
                "image": "qdrant/qdrant:latest",
                "container_name": container_name,
                "volumes": ["./qdrant_storage:/qdrant/storage"],
                "restart": "unless-stopped",
            }
        }
    }

    compose_path = project_dir / "docker-compose.yml"
    if compose_path.exists():
        print(f"  [skip] docker-compose.yml already exists", file=sys.stderr)
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
        print(f"  [created] docker-compose.yml", file=sys.stderr)

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
        print(f"  [created] .mcp.json", file=sys.stderr)

    # 4. Create RAG_data directory
    data_dir = project_dir / "RAG_data"
    data_dir.mkdir(exist_ok=True)
    print(f"  [created] RAG_data/", file=sys.stderr)

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
            print(f"  [updated] .gitignore", file=sys.stderr)
        else:
            print(f"  [skip] .gitignore already has entries", file=sys.stderr)
    else:
        with open(gitignore_path, "w") as f:
            f.write("# code-rag\n")
            for entry in gitignore_entries:
                f.write(f"{entry}\n")
        print(f"  [created] .gitignore", file=sys.stderr)

    # 6. Create .env template
    env_path = project_dir / ".env"
    if not env_path.exists():
        env_path.write_text(
            "# code-rag environment variables\n"
            "OPENROUTER_API_KEY=\n"
        )
        print(f"  [created] .env (add your OPENROUTER_API_KEY)", file=sys.stderr)

    # Print summary
    print(f"\nProject initialized at {project_dir}", file=sys.stderr)
    print(f"\nNext steps:", file=sys.stderr)
    print(f"  1. Add your OPENROUTER_API_KEY to .env", file=sys.stderr)
    print(f"  2. Drop files into RAG_data/ (code, markdown, or any supported format)", file=sys.stderr)
    print(f"  3. Update file_extensions in rag-config.json (e.g. [\".py\", \".md\", \".glsl\"])", file=sys.stderr)
    print(f"  4. Start Qdrant:  docker compose up -d", file=sys.stderr)
    print(f"  5. Run ingestion: code-rag ingest", file=sys.stderr)
    print(f"  6. (Optional) Add --enrich flag to generate LLM summaries per chunk", file=sys.stderr)
    print(f"  7. (Optional) Add domain-specific terms to key_terms in rag-config.json", file=sys.stderr)
    print(f"\nQdrant container: {container_name}", file=sys.stderr)
    print(f"Qdrant URL:       {qdrant_url}", file=sys.stderr)
    print(f"Collection:       {collection}", file=sys.stderr)


def cmd_ingest(args):
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


def cmd_serve(args):
    """Start the MCP server."""
    if args.project_dir:
        os.environ["RAG_PROJECT_DIR"] = str(Path(args.project_dir).resolve())
    from .server import main as server_main
    server_main()


def main():
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
        "--project-dir", type=str, default=".",
        help="Project directory (default: current directory)",
    )
    init_parser.add_argument(
        "--name", type=str, default=None,
        help="MCP server name (default: <dir-name>-rag)",
    )
    init_parser.add_argument(
        "--collection", type=str, default=None,
        help="Qdrant collection name (default: <dir_name>)",
    )

    # ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run the ingest pipeline (chunk, embed, index)",
    )
    ingest_parser.add_argument(
        "--project-dir", type=str, default=None,
        help="Project directory (default: current directory)",
    )
    ingest_parser.add_argument(
        "--no-recreate", action="store_true",
        help="Don't recreate the Qdrant collection (append to existing)",
    )
    ingest_parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Embedding batch size (default: 10)",
    )
    ingest_parser.add_argument(
        "--min-chunk-lines", type=int, default=None,
        help="Minimum chunk size in lines; smaller chunks merge into next sibling (default: from config or 20)",
    )
    ingest_parser.add_argument(
        "--max-chunk-lines", type=int, default=None,
        help="Maximum chunk size in lines; larger chunks split at paragraph boundaries (default: from config or 100)",
    )
    ingest_parser.add_argument(
        "--enrich", action="store_true", default=False,
        help="Run LLM enrichment on chunks (adds summaries and hypothetical questions)",
    )
    ingest_parser.add_argument(
        "--enrich-batch-size", type=int, default=None,
        help="Number of chunks per LLM enrichment call (default: 5)",
    )

    # serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the MCP server (for debugging; normally started by Claude Code)",
    )
    serve_parser.add_argument(
        "--project-dir", type=str, default=None,
        help="Project directory (default: current directory)",
    )

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)
