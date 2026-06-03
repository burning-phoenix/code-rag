"""
Naive fixed-window line chunker — the baseline for the AST-vs-naive ablation.

Splits every file into contiguous ``window`` -line blocks with no structural
awareness. Used only by the eval runner (``--chunker line-based``); it is never
registered in the production registry.
"""

from pathlib import Path

from ..chunkers import ChunkerRegistry
from ..config import ProjectConfig
from ..models import Chunk
from ..protocols import BaseChunker


class LineChunker(BaseChunker):
    """Splits files into fixed-size line windows, ignoring structure."""

    def __init__(self, window: int = 40) -> None:
        self._window = window

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


def build_line_registry(window: int = 40) -> ChunkerRegistry:
    """A registry that routes every file to the naive line chunker."""
    registry = ChunkerRegistry()
    registry.register_fallback(LineChunker(window))
    return registry
