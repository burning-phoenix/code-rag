"""
Pointer index: a concept → file/line-range mapping built at ingest time and
used for exact lookups (results come straight from the index, so they are
always real file locations). Each file type knows how to scan itself
(``BaseChunker.scan``); this module only runs the scan and handles
persistence and lookup.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from .chunkers import ChunkerRegistry
from .config import ProjectConfig

logger = logging.getLogger(__name__)


def build_index(
    data_dir: str | Path,
    config: ProjectConfig,
    registry: ChunkerRegistry,
) -> list[dict]:
    """
    Build the pointer index for all files matching ``config.file_extensions``.

    Dispatch is delegated to the registry: each chunker scans its own file type.
    Returns a list of index entry dicts.
    """
    data_dir = Path(data_dir)
    extensions = set(config.file_extensions)
    all_entries = []

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in extensions:
            continue
        chunker = registry.get(ext)
        if chunker is None:
            continue
        try:
            all_entries.extend(chunker.scan(file_path, config))
        except Exception as e:  # noqa: BLE001 — one bad file must not abort the batch
            logger.error("Failed to scan %s: %s", file_path, e)

    return [asdict(e) for e in all_entries]


def save_index(
    data_dir: str | Path,
    output_path: str | Path,
    config: ProjectConfig,
    registry: ChunkerRegistry,
) -> int:
    """Build and save the pointer index to a JSON file. Returns entry count."""
    index = build_index(data_dir, config, registry)
    output_path = Path(output_path)
    output_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return len(index)


def load_index(index_path: str | Path) -> list[dict]:
    """Load a previously saved pointer index."""
    data: list[dict] = json.loads(Path(index_path).read_text())
    return data


def search_index(index: list[dict], query: str) -> list[dict]:
    """
    Search the pointer index for entries matching a query.
    Uses case-insensitive substring matching (bidirectional).
    """
    query_lower = query.lower()
    results = []

    for entry in index:
        concept = entry["concept"].lower()
        if query_lower in concept or concept in query_lower:
            results.append(entry)

    # Sort: exact matches first, then by entry_type priority
    type_priority = {
        "definition": 0,
        "theorem": 1,
        "proposition": 2,
        "corollary": 3,
        "lemma": 4,
        "section": 5,
        "term": 6,
        "class": 7,
        "function": 8,
        "method": 9,
    }

    results.sort(
        key=lambda e: (
            0 if e["concept"].lower() == query_lower else 1,
            type_priority.get(e["entry_type"], 99),
        )
    )

    return results
