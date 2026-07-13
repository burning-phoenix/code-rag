"""
Chunker registry: maps file extensions to chunkers.

Adding support for a new file type means writing one ``BaseChunker`` subclass
and registering it in :func:`get_registry` — no existing dispatch code changes.
When two chunkers claim the same extension, the later registration wins.
"""

import logging
from pathlib import Path

from ..config import ProjectConfig
from ..models import Chunk
from ..protocols import BaseChunker
from .code import CodeChunker
from .lean import LeanChunker
from .markdown import MarkdownChunker
from .plaintext import PlaintextChunker
from .python import PythonChunker

logger = logging.getLogger(__name__)


class ChunkerRegistry:
    """Resolves the chunker responsible for a given file extension."""

    def __init__(self) -> None:
        self._chunkers: dict[str, BaseChunker] = {}  # ext → chunker
        self._fallback: BaseChunker | None = None

    def register(self, chunker: BaseChunker) -> None:
        """Register a chunker for each of its supported extensions."""
        for ext in chunker.supported_extensions():
            self._chunkers[ext] = chunker

    def register_fallback(self, chunker: BaseChunker) -> None:
        """Register the chunker used for any unclaimed extension."""
        self._fallback = chunker

    def get(self, extension: str) -> BaseChunker | None:
        """Return the chunker for an extension, or the fallback (may be ``None``)."""
        return self._chunkers.get(extension, self._fallback)


def get_registry() -> ChunkerRegistry:
    """Build and return the default chunker registry."""
    registry = ChunkerRegistry()
    registry.register(MarkdownChunker())
    registry.register(CodeChunker())
    registry.register(PythonChunker())  # claims .py from CodeChunker (stdlib ast)
    registry.register(LeanChunker())
    registry.register_fallback(PlaintextChunker())
    return registry


def chunk_directory(
    directory: str | Path,
    config: ProjectConfig,
    registry: ChunkerRegistry,
) -> list[Chunk]:
    """Chunk every file matching ``config.file_extensions`` (recursively)."""
    directory = Path(directory)
    extensions = set(config.file_extensions)
    all_chunks: list[Chunk] = []

    for file_path in sorted(directory.rglob("*")):
        if not (file_path.is_file() and file_path.suffix.lower() in extensions):
            continue
        chunker = registry.get(file_path.suffix.lower())
        if chunker is None:
            continue
        try:
            file_chunks = chunker.chunk(file_path, config)
        except Exception as e:  # noqa: BLE001 — one bad file must not abort the batch
            logger.error("Failed to chunk %s: %s", file_path, e)
            continue
        # Stamp the corpus-relative path so chunk identity is collision-free
        # across same-named files in different directories. Chunkers never see
        # directory layout; this function is the one place that knows where
        # files live.
        rel_path = file_path.relative_to(directory).as_posix()
        for chunk in file_chunks:
            chunk.source_path = rel_path
        all_chunks.extend(file_chunks)

    return all_chunks


__all__ = ["ChunkerRegistry", "get_registry", "chunk_directory"]
