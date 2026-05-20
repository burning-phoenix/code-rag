"""
Chunk router: dispatches files to the appropriate chunker based on extension.

- .md files → markdown chunker (existing, unchanged)
- Code files (known extensions) → AST-aware code chunker
- Other files → plaintext fallback chunker
"""

import logging
from pathlib import Path

from .chunker import Chunk, chunk_file as markdown_chunk_file
from .code_chunker import chunk_code_file, EXTENSION_TO_LANGUAGE
from .lean_chunker import chunk_lean_file
from .plaintext_chunker import chunk_plaintext_file
from .config import ProjectConfig

logger = logging.getLogger(__name__)


def chunk_file(file_path: Path, config: ProjectConfig) -> list[Chunk]:
    """Route a single file to the appropriate chunker based on its extension."""
    ext = file_path.suffix.lower()

    if ext == ".md":
        return markdown_chunk_file(file_path, config.min_chunk_lines, config.max_chunk_lines)
    if ext == ".lean":
        return chunk_lean_file(file_path)

    language = EXTENSION_TO_LANGUAGE.get(ext)
    if language:
        return chunk_code_file(
            file_path, language,
            min_chunk_lines=config.code_chunk_min_lines,
            max_chunk_lines=config.code_chunk_max_lines,
        )

    return chunk_plaintext_file(file_path, config.min_chunk_lines, config.max_chunk_lines)


def chunk_directory(directory: str | Path, config: ProjectConfig) -> list[Chunk]:
    """Chunk all files matching config.file_extensions in a directory (recursively)."""
    directory = Path(directory)
    extensions = set(config.file_extensions)
    all_chunks: list[Chunk] = []

    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                file_chunks = chunk_file(file_path, config)
                all_chunks.extend(file_chunks)
            except Exception as e:
                logger.error("Failed to chunk %s: %s", file_path, e)

    return all_chunks
