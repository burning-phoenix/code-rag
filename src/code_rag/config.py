"""
Project configuration: loads rag-config.json and resolves all paths.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class ProjectConfig:
    """All settings for a code-rag project instance."""

    name: str
    collection: str
    data_dir: Path
    pointer_index: Path
    description: str
    key_terms: list[str]
    qdrant_url: str
    openrouter_api_key: str
    min_chunk_lines: int = 20
    max_chunk_lines: int = 100
    # --- Code support fields ---
    file_extensions: list[str] = field(default_factory=lambda: [".md"])
    enrich_chunks: bool = False
    enrichment_model: str = "meta-llama/llama-4-scout"
    code_chunk_min_lines: int = 5
    code_chunk_max_lines: int = 150


_cached_config: ProjectConfig | None = None


def _resolve_project_dir() -> Path:
    """Determine the project directory from env var or cwd."""
    env_dir = os.getenv("RAG_PROJECT_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return Path.cwd()


def load_config(project_dir: Path | str | None = None) -> ProjectConfig:
    """
    Load project configuration.

    Resolution order:
    1. Explicit project_dir argument
    2. RAG_PROJECT_DIR environment variable
    3. Current working directory

    The project directory must contain a rag-config.json file.
    A .env file in the project directory is loaded if present.
    Environment variables QDRANT_URL and OPENROUTER_API_KEY override config values.
    """
    global _cached_config
    if _cached_config is not None and project_dir is None:
        return _cached_config

    project_dir = Path(project_dir).resolve() if project_dir is not None else _resolve_project_dir()

    # Load .env from project directory if it exists
    env_path = project_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Load config file
    config_path = project_dir / "rag-config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No rag-config.json found in {project_dir}.\n"
            f"Run 'code-rag init' in your project directory first."
        )

    with open(config_path) as f:
        raw = json.load(f)

    # Resolve relative paths against project directory
    data_dir = project_dir / raw.get("data_dir", "RAG_data")
    pointer_index = project_dir / raw.get("pointer_index", "pointer_index.json")

    # Build config with env var overrides
    config = ProjectConfig(
        name=raw.get("name", "code-rag"),
        collection=raw.get("collection", "documents"),
        data_dir=data_dir,
        pointer_index=pointer_index,
        description=raw.get("description", "A local RAG server for document search."),
        key_terms=raw.get("key_terms", []),
        qdrant_url=os.getenv("QDRANT_URL", raw.get("qdrant_url", "")),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        min_chunk_lines=raw.get("min_chunk_lines", 20),
        max_chunk_lines=raw.get("max_chunk_lines", 100),
        file_extensions=raw.get("file_extensions", [".md"]),
        enrich_chunks=raw.get("enrich_chunks", False),
        enrichment_model=raw.get("enrichment_model", "meta-llama/llama-4-scout"),
        code_chunk_min_lines=raw.get("code_chunk_min_lines", 5),
        code_chunk_max_lines=raw.get("code_chunk_max_lines", 150),
    )

    if project_dir is None or project_dir == _resolve_project_dir():
        _cached_config = config

    return config


def reset_config() -> None:
    """Clear cached config (useful for testing)."""
    global _cached_config
    _cached_config = None
