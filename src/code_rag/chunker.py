"""
Markdown chunker: splits documents by section headers while preserving
mathematical formulas and their surrounding context.

Post-processing enforces min/max chunk sizes:
- Small chunks (< min_chunk_lines) are merged into the next sibling
- Large chunks (> max_chunk_lines) are split at paragraph boundaries
"""

from dataclasses import dataclass
import re
from pathlib import Path


@dataclass
class Chunk:
    """A logical section of a document (markdown, code, or plaintext)."""

    text: str
    file_name: str
    section_title: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    math_density: float = 0.0  # ratio of math-containing lines to total lines
    # --- Code support fields ---
    chunk_id: str | None = None        # stable UUID, assigned at creation (for parent-child links)
    chunk_type: str = "markdown"       # "markdown" | "code" | "plaintext"
    language: str | None = None        # programming language identifier (e.g. "python", "glsl")
    symbol_name: str | None = None     # function/class/method name
    symbol_type: str | None = None     # "function" | "class" | "method" | "module_preamble"
    parent_id: str | None = None       # chunk_id of parent chunk (class → method link)
    # --- Enrichment fields (populated optionally at ingest time) ---
    summary: str | None = None         # LLM-generated summary
    hypothetical_questions: list[str] | None = None  # LLM-generated questions this chunk answers


def _compute_math_density(text: str) -> float:
    """Compute the fraction of lines that contain mathematical notation."""
    lines = text.split("\n")
    if not lines:
        return 0.0

    math_lines = 0
    in_display_math = False

    for line in lines:
        stripped = line.strip()

        # Track display math blocks ($$...$$)
        if stripped.startswith("$$"):
            in_display_math = not in_display_math
            math_lines += 1
            continue

        if in_display_math:
            math_lines += 1
            continue

        # Inline math ($...$) — at least one pair
        if re.search(r"(?<!\$)\$(?!\$).+?\$(?!\$)", stripped):
            math_lines += 1
            continue

        # Lines with common math markers
        if any(marker in stripped for marker in [
            "\\\\", "\\frac", "\\sum", "\\prod", "\\int",
            "\\mathbb", "\\mathcal", "\\operatorname",
            "\\oplus", "\\odot", "\\oslash", "\\otimes",
            "\\in", "\\to", "\\mapsto", "\\coloneqq",
            "\\leq", "\\geq", "\\neq", "\\subseteq",
            "^{", "_{",
        ]):
            math_lines += 1

    return math_lines / len(lines)


def _is_section_header(line: str) -> bool:
    """Check if line is a markdown section header (## or deeper)."""
    return bool(re.match(r"^#{1,6}\s+", line.strip()))


def _extract_header_title(line: str) -> str:
    """Extract the title text from a markdown header line."""
    match = re.match(r"^#{1,6}\s+(.*)", line.strip())
    return match.group(1) if match else ""


def _chunk_num_lines(chunk: Chunk) -> int:
    """Number of lines in a chunk."""
    return chunk.end_line - chunk.start_line + 1


def _merge_small_chunks(chunks: list[Chunk], min_lines: int) -> list[Chunk]:
    """Merge chunks smaller than min_lines into adjacent siblings.

    Small chunks are absorbed forward into the next chunk. If the last chunk
    is small, it's absorbed backward into the previous chunk. Section title
    of the earlier chunk is preserved so the header context isn't lost.
    """
    if not chunks or min_lines <= 0:
        return chunks

    result: list[Chunk] = []

    for chunk in chunks:
        if result and _chunk_num_lines(result[-1]) < min_lines:
            # Previous chunk was too small — merge it into current chunk
            prev = result[-1]
            merged_text = prev.text + "\n\n" + chunk.text
            result[-1] = Chunk(
                text=merged_text.strip(),
                file_name=chunk.file_name,
                section_title=prev.section_title,
                start_line=prev.start_line,
                end_line=chunk.end_line,
                math_density=_compute_math_density(merged_text),
            )
        else:
            result.append(chunk)

    # If the last chunk is still too small, merge backward into previous
    if len(result) > 1 and _chunk_num_lines(result[-1]) < min_lines:
        last = result.pop()
        prev = result[-1]
        merged_text = prev.text + "\n\n" + last.text
        result[-1] = Chunk(
            text=merged_text.strip(),
            file_name=last.file_name,
            section_title=prev.section_title,
            start_line=prev.start_line,
            end_line=last.end_line,
            math_density=_compute_math_density(merged_text),
        )

    return result


def _split_large_chunk(chunk: Chunk, max_lines: int) -> list[Chunk]:
    """Split a chunk larger than max_lines at paragraph boundaries.

    Finds blank lines that aren't inside display math blocks ($$...$$)
    and splits greedily. Continuation sub-chunks get the section title
    prepended so embeddings retain context.
    """
    lines = chunk.text.split("\n")
    if len(lines) <= max_lines:
        return [chunk]

    # Find safe split points: blank lines outside display math blocks
    in_math = False
    safe_blanks = []
    for i, line in enumerate(lines):
        if line.strip().startswith("$$"):
            in_math = not in_math
        if not line.strip() and not in_math:
            safe_blanks.append(i)

    if not safe_blanks:
        # No paragraph breaks — can't split meaningfully
        return [chunk]

    # Greedy split: accumulate until max_lines, split at last safe blank
    result = []
    part_start = 0
    part_num = 0

    while part_start < len(lines):
        part_end = min(part_start + max_lines, len(lines))

        if part_end >= len(lines):
            # Remaining lines fit within max — take them all
            break

        # Find the last safe blank at or before part_end
        split_at = None
        for sp in safe_blanks:
            if part_start < sp <= part_end:
                split_at = sp

        if split_at is None:
            # No safe blank within range — find the next one after part_end
            for sp in safe_blanks:
                if sp > part_start:
                    split_at = sp
                    break

        if split_at is None:
            # No more safe blanks — take everything remaining
            break

        part_text = "\n".join(lines[part_start:split_at]).strip()
        if part_text:
            if part_num > 0:
                part_text = f"## {chunk.section_title} (continued)\n\n{part_text}"
            result.append(Chunk(
                text=part_text,
                file_name=chunk.file_name,
                section_title=chunk.section_title,
                start_line=chunk.start_line + part_start,
                end_line=chunk.start_line + split_at - 1,
                math_density=_compute_math_density(part_text),
            ))
            part_num += 1

        part_start = split_at + 1  # skip the blank line

    # Flush remaining lines
    if part_start < len(lines):
        part_text = "\n".join(lines[part_start:]).strip()
        if part_text:
            if part_num > 0:
                part_text = f"## {chunk.section_title} (continued)\n\n{part_text}"
            result.append(Chunk(
                text=part_text,
                file_name=chunk.file_name,
                section_title=chunk.section_title,
                start_line=chunk.start_line + part_start,
                end_line=chunk.end_line,
                math_density=_compute_math_density(part_text),
            ))

    return result if result else [chunk]


def _split_large_chunks(chunks: list[Chunk], max_lines: int) -> list[Chunk]:
    """Apply large-chunk splitting across a list of chunks."""
    if max_lines <= 0:
        return chunks

    result = []
    for chunk in chunks:
        if _chunk_num_lines(chunk) > max_lines:
            result.extend(_split_large_chunk(chunk, max_lines))
        else:
            result.append(chunk)
    return result


def chunk_file(
    file_path: str | Path,
    min_chunk_lines: int = 20,
    max_chunk_lines: int = 100,
) -> list[Chunk]:
    """
    Chunk a markdown file by section headers with size enforcement.

    1. Splits on any markdown header (# through ######)
    2. Merges chunks < min_chunk_lines into next sibling
    3. Splits chunks > max_chunk_lines at paragraph boundaries

    Set min_chunk_lines=0 or max_chunk_lines=0 to disable that pass.
    """
    file_path = Path(file_path)
    file_name = file_path.name

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return []

    # --- Pass 1: split on section headers ---
    chunks: list[Chunk] = []
    current_title = file_name  # default title for content before first header
    current_lines: list[str] = []
    current_start = 1  # 1-indexed

    for i, line in enumerate(lines, start=1):
        if _is_section_header(line):
            # Flush previous chunk if it has content
            if current_lines:
                text = "".join(current_lines).strip()
                if text:
                    chunks.append(Chunk(
                        text=text,
                        file_name=file_name,
                        section_title=current_title,
                        start_line=current_start,
                        end_line=i - 1,
                        math_density=_compute_math_density(text),
                    ))
            current_title = _extract_header_title(line)
            current_lines = [line]
            current_start = i
        else:
            current_lines.append(line)

    # Flush final chunk
    if current_lines:
        text = "".join(current_lines).strip()
        if text:
            chunks.append(Chunk(
                text=text,
                file_name=file_name,
                section_title=current_title,
                start_line=current_start,
                end_line=len(lines),
                math_density=_compute_math_density(text),
            ))

    # --- Pass 2: merge small chunks (before splitting) ---
    if min_chunk_lines > 0:
        chunks = _merge_small_chunks(chunks, min_chunk_lines)

    # --- Pass 3: split large chunks ---
    if max_chunk_lines > 0:
        chunks = _split_large_chunks(chunks, max_chunk_lines)

    return chunks


def chunk_directory(
    directory: str | Path,
    min_chunk_lines: int = 20,
    max_chunk_lines: int = 100,
) -> list[Chunk]:
    """Chunk all .md files in a directory (recursively) with size enforcement."""
    directory = Path(directory)
    all_chunks: list[Chunk] = []

    for md_file in sorted(directory.rglob("*.md")):
        all_chunks.extend(chunk_file(md_file, min_chunk_lines, max_chunk_lines))

    return all_chunks
