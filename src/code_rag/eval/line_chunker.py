"""
Fixed-window line chunker — the baseline the structure-aware chunkers are
compared against.

Splits every file into contiguous ``window``-line blocks with no knowledge of
the file's structure. Used only by the eval runner (``--chunker line-based``);
it is never registered in the production registry.
"""

from pathlib import Path

from ..chunkers import ChunkerRegistry
from ..config import ProjectConfig
from ..models import Chunk
from ..protocols import BaseChunker

# A window no file exceeds: one chunk per file. This is the whole-file
# configuration used to test the metrics themselves — it maximises coverage
# while returning mostly irrelevant text, so a working metric suite must never
# rank it best overall (see canary_check in eval/report.py).
WHOLE_FILE_WINDOW = 10**9

# Embedding-input cap for whole-file chunks. A full paper is far past the
# embedding model's token limit (a 10-chunk batch of whole files crashed the
# 2026-07-09 run), so whole-file chunks embed a truncated prefix — about 3k
# tokens, putting a batch on the same scale as the enriched batches that
# succeed. The chunk's *line range* stays whole-file, which is all the metrics
# read; this configuration exists to test the metrics' geometry handling, so
# the quality of its embeddings was never the point.
WHOLE_FILE_EMBED_CHARS = 12_000


class LineChunker(BaseChunker):
    """Splits files into fixed-size line windows, ignoring structure."""

    def __init__(self, window: int = 40, max_chars: int | None = None) -> None:
        self._window = window
        self._max_chars = max_chars

    def supported_extensions(self) -> set[str]:
        return set()  # registered as the fallback for every extension

    def chunk(self, file_path: Path, config: ProjectConfig) -> list[Chunk]:
        file_path = Path(file_path)
        file_name = file_path.name
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if not lines:
            return []

        chunks: list[Chunk] = []
        for start in range(0, len(lines), self._window):
            block = lines[start : start + self._window]
            text = "".join(block).strip()
            if not text:
                continue
            if self._max_chars is not None and len(text) > self._max_chars:
                text = text[: self._max_chars]
            chunks.append(
                Chunk(
                    text=text,
                    file_name=file_name,
                    section_title=f"{file_name}:{start + 1}",
                    start_line=start + 1,
                    end_line=start + len(block),
                    chunk_type="plaintext",
                )
            )
        return chunks


def build_line_registry(window: int = 40, max_chars: int | None = None) -> ChunkerRegistry:
    """A registry that routes every file to the fixed-window line chunker."""
    registry = ChunkerRegistry()
    registry.register_fallback(LineChunker(window, max_chars))
    return registry
