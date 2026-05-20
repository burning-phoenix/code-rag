"""
Lean chunker: lightweight top-level declaration chunking for Lean source files.

This avoids introducing a tree-sitter Lean dependency while still exposing
useful declaration-level chunks for semantic search.
"""

import re
import uuid
from pathlib import Path

from .chunker import Chunk

_LEAN_DECL_PATTERN = re.compile(
    r"^\s*"
    r"(?:(?:private|protected|noncomputable|unsafe|partial|scoped|local)\s+)*"
    r"(def|theorem|lemma|structure|class|inductive|abbrev|instance|axiom|example)\s+"
    r"([^\s(:{]+)"
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
    """Map Lean declaration kinds onto existing code-rag symbol categories."""
    if kind in {"theorem", "lemma"}:
        return kind
    if kind in {"structure", "class", "inductive"}:
        return "class"
    if kind in {"def", "abbrev", "axiom"}:
        return "function"
    return kind


def chunk_lean_file(file_path: str | Path) -> list[Chunk]:
    """Chunk a Lean file into a module preamble plus top-level declarations."""
    file_path = Path(file_path)
    file_name = file_path.name

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return []

    declarations: list[tuple[int, str, str]] = []
    depth = 0

    for line_no, line in enumerate(lines, start=1):
        code, depth = _strip_comments(line, depth)
        match = _LEAN_DECL_PATTERN.match(code)
        if match:
            declarations.append((line_no, match.group(1), match.group(2)))

    if not declarations:
        text = "".join(lines).strip()
        if not text:
            return []
        return [Chunk(
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
        )]

    chunks: list[Chunk] = []

    first_decl_line = declarations[0][0]
    if first_decl_line > 1:
        preamble_text = "".join(lines[: first_decl_line - 1]).strip()
        if preamble_text:
            chunks.append(Chunk(
                text=preamble_text,
                file_name=file_name,
                section_title=f"{file_name} (module preamble)",
                start_line=1,
                end_line=first_decl_line - 1,
                chunk_id=str(uuid.uuid4()),
                chunk_type="code",
                language="lean",
                symbol_name=f"{file_name} preamble",
                symbol_type="module_preamble",
            ))

    for idx, (start_line, kind, name) in enumerate(declarations):
        end_line = declarations[idx + 1][0] - 1 if idx + 1 < len(declarations) else len(lines)
        text = "".join(lines[start_line - 1 : end_line]).strip()
        if not text:
            continue

        chunks.append(Chunk(
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
        ))

    return chunks
