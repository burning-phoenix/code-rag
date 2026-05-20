"""
Plaintext chunker: fallback line-based chunker for files without
AST support or markdown structure. Splits at blank-line paragraph
boundaries with min/max size enforcement.
"""

import uuid
from pathlib import Path

from .chunker import Chunk


def chunk_plaintext_file(
    file_path: str | Path,
    min_chunk_lines: int = 20,
    max_chunk_lines: int = 100,
) -> list[Chunk]:
    """
    Chunk a plaintext file at blank-line paragraph boundaries.

    Merges small paragraphs and splits overly long sections, similar to
    the markdown chunker but without header awareness.
    """
    file_path = Path(file_path)
    file_name = file_path.name

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return []

    # Split into paragraphs at blank lines
    paragraphs: list[tuple[int, int, str]] = []  # (start_line, end_line, text)
    para_start = 1
    current_lines: list[str] = []

    for i, line in enumerate(lines, start=1):
        if line.strip() == "":
            if current_lines:
                text = "".join(current_lines).strip()
                if text:
                    paragraphs.append((para_start, i - 1, text))
                current_lines = []
            para_start = i + 1
        else:
            if not current_lines:
                para_start = i
            current_lines.append(line)

    # Flush final paragraph
    if current_lines:
        text = "".join(current_lines).strip()
        if text:
            paragraphs.append((para_start, len(lines), text))

    if not paragraphs:
        return []

    # Merge small paragraphs and build chunks
    chunks: list[Chunk] = []
    acc_text = ""
    acc_start = paragraphs[0][0]
    acc_end = paragraphs[0][1]

    for start, end, text in paragraphs:
        candidate = (acc_text + "\n\n" + text).strip() if acc_text else text
        candidate_lines = end - acc_start + 1

        if candidate_lines > max_chunk_lines > 0 and acc_text:
            # Current accumulator is big enough, flush it
            chunks.append(Chunk(
                text=acc_text,
                file_name=file_name,
                section_title=file_name,
                start_line=acc_start,
                end_line=acc_end,
                chunk_id=str(uuid.uuid4()),
                chunk_type="plaintext",
            ))
            acc_text = text
            acc_start = start
            acc_end = end
        else:
            acc_text = candidate
            acc_end = end
            if not acc_text.startswith(text):
                pass  # acc_start stays the same
            else:
                acc_start = start

    # Flush remaining
    if acc_text:
        chunks.append(Chunk(
            text=acc_text,
            file_name=file_name,
            section_title=file_name,
            start_line=acc_start,
            end_line=acc_end,
            chunk_id=str(uuid.uuid4()),
            chunk_type="plaintext",
        ))

    # Merge chunks that are still too small
    if min_chunk_lines > 0 and len(chunks) > 1:
        merged: list[Chunk] = [chunks[0]]
        for chunk in chunks[1:]:
            prev = merged[-1]
            prev_lines = prev.end_line - prev.start_line + 1
            if prev_lines < min_chunk_lines:
                combined_text = prev.text + "\n\n" + chunk.text
                merged[-1] = Chunk(
                    text=combined_text.strip(),
                    file_name=file_name,
                    section_title=file_name,
                    start_line=prev.start_line,
                    end_line=chunk.end_line,
                    chunk_id=prev.chunk_id,
                    chunk_type="plaintext",
                )
            else:
                merged.append(chunk)
        chunks = merged

    return chunks
