"""
Lean chunker: lightweight top-level declaration chunking for Lean source files.

This avoids introducing a tree-sitter Lean dependency while still exposing
useful declaration-level chunks for semantic search. The declaration regex and
comment stripper live here only (previously duplicated in the pointer index).
"""

import re
import uuid
from pathlib import Path

from ..config import ProjectConfig
from ..models import Chunk, IndexEntry
from ..protocols import BaseChunker
from ._shared import add_term_entries

_LEAN_DECL_PATTERN = re.compile(
    r"^\s*"
    r"(?:(?:private|protected|noncomputable|unsafe|partial|scoped|local)\s+)*"
    r"(def|theorem|lemma|structure|class|inductive|abbrev|instance|axiom|example)\s+"
    r"([^\s(:{]+)"
)

_ATTR_PATTERN = re.compile(r"^\s*@\[")
_MODIFIER_PATTERN = re.compile(
    r"^\s*(?:private|protected|noncomputable|unsafe|partial|scoped|local)\b"
)


def _strip_comments(line: str, depth: int) -> tuple[str, int]:
    """Remove Lean line/block comments while tracking nested block depth."""
    out: list[str] = []
    i = 0

    while i < len(line):
        if depth == 0 and line.startswith("--", i):
            break
        if line.startswith("/-", i):
            depth += 1
            i += 2
            continue
        if depth > 0 and line.startswith("-/", i):
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(line[i])
        i += 1

    return "".join(out), depth


def _symbol_type(kind: str) -> str:
    """Map Lean declaration kinds onto code-rag symbol categories (for chunks)."""
    if kind in {"theorem", "lemma"}:
        return kind
    if kind in {"structure", "class", "inductive"}:
        return "class"
    if kind in {"def", "abbrev", "axiom"}:
        return "function"
    return kind


def _entry_type(kind: str) -> str:
    """Map Lean declarations onto pointer-index entry types (for scans)."""
    if kind == "def":
        return "definition"
    if kind in {"theorem", "lemma"}:
        return kind
    if kind in {"structure", "class", "inductive"}:
        return "class"
    if kind == "abbrev":
        return "definition"
    return kind


def _find_declarations(lines: list[str]) -> list[tuple[int, str, str]]:
    """Return (line_no, kind, name) for each top-level declaration, comments stripped."""
    declarations: list[tuple[int, str, str]] = []
    depth = 0
    for line_no, line in enumerate(lines, start=1):
        code, depth = _strip_comments(line, depth)
        match = _LEAN_DECL_PATTERN.match(code)
        if match:
            declarations.append((line_no, match.group(1), match.group(2)))
    return declarations


def _leading_block_start(lines: list[str], decl_line: int) -> int:
    """1-indexed start of a declaration's leading doc/attribute block.

    Walks upward from the declaration keyword line absorbing the contiguous run
    of doc comments (``/-- … -/``, possibly multi-line), line comments, attribute
    lines (``@[…]``) and bare modifier lines, stopping at a blank line or code.
    This keeps each declaration's docstring with the declaration it describes,
    instead of leaking it into the previous chunk.
    """
    start = decl_line
    i = decl_line - 1  # 1-indexed line just above the keyword
    while i >= 1:
        stripped = lines[i - 1].strip()
        if stripped == "":
            break
        if _ATTR_PATTERN.match(lines[i - 1]) or _MODIFIER_PATTERN.match(lines[i - 1]):
            start = i
            i -= 1
            continue
        if stripped.endswith("-/"):
            # End of a block comment — walk up to its opener and absorb it whole.
            j = i
            while j >= 1 and not lines[j - 1].lstrip().startswith("/-"):
                j -= 1
            if j >= 1:
                start = j
                i = j - 1
                continue
            break
        if stripped.startswith("--"):
            start = i
            i -= 1
            continue
        break  # reached code (previous declaration's body)
    return start


class LeanChunker(BaseChunker):
    """Chunks and scans Lean files by top-level declaration."""

    def supported_extensions(self) -> set[str]:
        return {".lean"}

    def chunk(self, file_path: Path, config: ProjectConfig) -> list[Chunk]:
        """Chunk a Lean file into a module preamble plus top-level declarations."""
        file_path = Path(file_path)
        file_name = file_path.name

        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if not lines:
            return []

        declarations = _find_declarations(lines)

        if not declarations:
            text = "".join(lines).strip()
            if not text:
                return []
            return [
                Chunk(
                    text=text,
                    file_name=file_name,
                    section_title=f"{file_name} (module preamble)",
                    start_line=1,
                    end_line=len(lines),
                    chunk_id=str(uuid.uuid4()),
                    chunk_type="code",
                    language="lean",
                    symbol_name=f"{file_name} preamble",
                    symbol_type="module_preamble",
                )
            ]

        chunks: list[Chunk] = []

        # Each declaration's chunk starts at its leading doc/attribute block, so a
        # docstring travels with the declaration it documents (not the prior one).
        doc_starts = [_leading_block_start(lines, kw_line) for kw_line, _, _ in declarations]

        if doc_starts[0] > 1:
            preamble_text = "".join(lines[: doc_starts[0] - 1]).strip()
            if preamble_text:
                chunks.append(
                    Chunk(
                        text=preamble_text,
                        file_name=file_name,
                        section_title=f"{file_name} (module preamble)",
                        start_line=1,
                        end_line=doc_starts[0] - 1,
                        chunk_id=str(uuid.uuid4()),
                        chunk_type="code",
                        language="lean",
                        symbol_name=f"{file_name} preamble",
                        symbol_type="module_preamble",
                    )
                )

        for idx, (_kw_line, kind, name) in enumerate(declarations):
            start_line = doc_starts[idx]
            end_line = doc_starts[idx + 1] - 1 if idx + 1 < len(declarations) else len(lines)
            text = "".join(lines[start_line - 1 : end_line]).strip()
            if not text:
                continue

            chunks.append(
                Chunk(
                    text=text,
                    file_name=file_name,
                    section_title=name,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_id=str(uuid.uuid4()),
                    chunk_type="code",
                    language="lean",
                    symbol_name=name,
                    symbol_type=_symbol_type(kind),
                )
            )

        return chunks

    def scan(self, file_path: Path, config: ProjectConfig) -> list[IndexEntry]:
        """Scan a Lean file for top-level declarations and configured key terms."""
        entries: list[IndexEntry] = []
        file_path = Path(file_path)
        file_name = file_path.name
        abs_path = str(file_path.resolve())

        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        declarations = _find_declarations(lines)
        doc_starts = [_leading_block_start(lines, kw_line) for kw_line, _, _ in declarations]

        for idx, (start_line, kind, name) in enumerate(declarations):
            # Entry points at the declaration keyword; bound the range before the
            # next declaration's leading doc block so it doesn't overrun.
            end_line = doc_starts[idx + 1] - 1 if idx + 1 < len(declarations) else len(lines)
            entries.append(
                IndexEntry(
                    concept=name,
                    file_name=file_name,
                    file_path=abs_path,
                    start_line=start_line,
                    end_line=end_line,
                    entry_type=_entry_type(kind),
                )
            )

        if config.key_terms:
            add_term_entries(entries, lines, file_name, abs_path, config.key_terms)

        return entries
