"""
Python chunker built on the standard-library :mod:`ast`.

Python's own parser gives precise, structural boundaries that the generic
tree-sitter chunker can't match cheaply: every method (regardless of class
size), decorators, and exact line spans. We use it for ``.py`` and fall back to
the tree-sitter :class:`~code_rag.chunkers.code.CodeChunker` on ``SyntaxError``
(malformed or Python-2 files), which is error-tolerant.

Granularity: a class yields one parent chunk (full class, for context) plus a
child chunk per *substantial* method (span >= ``code_chunk_min_lines``); trivial
one-liners stay inside the parent. Large top-level functions are split at body
statement boundaries with the signature prepended.
"""

import ast
import logging
import textwrap
import uuid
from pathlib import Path

from ..config import ProjectConfig
from ..models import Chunk, IndexEntry
from ..protocols import BaseChunker
from .code import CodeChunker

logger = logging.getLogger(__name__)

_FuncDef = (ast.FunctionDef, ast.AsyncFunctionDef)


def _end(node: ast.AST) -> int:
    """End line of a node (``end_lineno`` is Optional in the AST)."""
    end = getattr(node, "end_lineno", None)
    return int(end) if end is not None else int(node.lineno)  # type: ignore[attr-defined]


def _start(node: ast.AST) -> int:
    """Start line of a node, including any leading decorators."""
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        return int(min(d.lineno for d in decorators))
    return int(node.lineno)  # type: ignore[attr-defined]


def _segment(lines: list[str], start: int, end: int) -> str:
    """Exact source for an inclusive 1-indexed line range, de-indented."""
    return textwrap.dedent("".join(lines[start - 1 : end])).strip("\n")


class PythonChunker(BaseChunker):
    """AST chunker for Python using the standard-library parser."""

    def supported_extensions(self) -> set[str]:
        return {".py"}

    def chunk(self, file_path: Path, config: ProjectConfig) -> list[Chunk]:
        file_path = Path(file_path)
        file_name = file_path.name
        source = file_path.read_text(encoding="utf-8", errors="replace")
        if not source.strip():
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning(
                "ast.parse failed for %s (%s); falling back to tree-sitter", file_name, e
            )
            return CodeChunker().chunk(file_path, config)

        lines = source.splitlines(keepends=True)
        min_lines = config.code_chunk_min_lines
        max_lines = config.code_chunk_max_lines

        chunks: list[Chunk] = []
        preamble: list[tuple[int, int]] = []

        def flush_preamble() -> None:
            if not preamble:
                return
            start, end = preamble[0][0], preamble[-1][1]
            text = _segment(lines, start, end)
            if text:
                chunks.append(
                    Chunk(
                        text=text,
                        file_name=file_name,
                        section_title=f"{file_name} (module preamble)",
                        start_line=start,
                        end_line=end,
                        chunk_id=str(uuid.uuid4()),
                        chunk_type="code",
                        language="python",
                        symbol_name=f"{file_name} preamble",
                        symbol_type="module_preamble",
                    )
                )
            preamble.clear()

        for node in tree.body:
            if isinstance(node, _FuncDef):
                flush_preamble()
                chunks.extend(self._function_chunks(node, lines, file_name, max_lines))
            elif isinstance(node, ast.ClassDef):
                flush_preamble()
                chunks.extend(self._class_chunks(node, lines, file_name, min_lines))
            else:
                preamble.append((node.lineno, _end(node)))

        flush_preamble()
        return chunks

    def _mk(
        self,
        *,
        text: str,
        file_name: str,
        symbol_name: str,
        symbol_type: str,
        start: int,
        end: int,
        chunk_id: str | None = None,
        parent_id: str | None = None,
        section_title: str | None = None,
    ) -> Chunk:
        return Chunk(
            text=text,
            file_name=file_name,
            section_title=section_title or symbol_name,
            start_line=start,
            end_line=end,
            chunk_id=chunk_id or str(uuid.uuid4()),
            chunk_type="code",
            language="python",
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            parent_id=parent_id,
        )

    def _function_chunks(
        self, node: ast.AST, lines: list[str], file_name: str, max_lines: int
    ) -> list[Chunk]:
        start, end = _start(node), _end(node)
        name = node.name  # type: ignore[attr-defined]
        if end - start + 1 <= max_lines or not node.body:  # type: ignore[attr-defined]
            return [
                self._mk(
                    text=_segment(lines, start, end),
                    file_name=file_name,
                    symbol_name=name,
                    symbol_type="function",
                    start=start,
                    end=end,
                )
            ]
        return self._split_function(node, lines, file_name, max_lines)

    def _split_function(
        self, node: ast.AST, lines: list[str], file_name: str, max_lines: int
    ) -> list[Chunk]:
        """Split an oversized function at top-level body-statement boundaries.

        Each part carries the function signature for context; line ranges stay
        disjoint so point identities don't collide.
        """
        name = node.name  # type: ignore[attr-defined]
        body = node.body  # type: ignore[attr-defined]
        fn_start = _start(node)
        sig_text = (
            _segment(lines, fn_start, body[0].lineno - 1) if body[0].lineno > fn_start else ""
        )

        chunks: list[Chunk] = []
        group: list[ast.stmt] = []
        group_lines = 0
        part = 0

        def emit() -> None:
            nonlocal part, group, group_lines
            g_start, g_end = _start(group[0]), _end(group[-1])
            if part == 0:
                text = _segment(lines, fn_start, g_end)
                start_line = fn_start
            else:
                body_text = _segment(lines, g_start, g_end)
                text = f"{sig_text}\n{body_text}" if sig_text else body_text
                start_line = g_start
            chunks.append(
                self._mk(
                    text=text,
                    file_name=file_name,
                    symbol_name=name,
                    symbol_type="function",
                    start=start_line,
                    end=g_end,
                    section_title=f"{name} (part {part + 1})",
                )
            )
            part += 1
            group = []
            group_lines = 0

        for stmt in body:
            span = _end(stmt) - _start(stmt) + 1
            if group and group_lines + span > max_lines:
                emit()
            group.append(stmt)
            group_lines += span
        if group:
            emit()
        return chunks

    def _class_chunks(
        self, node: ast.ClassDef, lines: list[str], file_name: str, min_lines: int
    ) -> list[Chunk]:
        c_start, c_end = _start(node), _end(node)
        class_id = str(uuid.uuid4())
        chunks = [
            self._mk(
                text=_segment(lines, c_start, c_end),
                file_name=file_name,
                symbol_name=node.name,
                symbol_type="class",
                start=c_start,
                end=c_end,
                chunk_id=class_id,
            )
        ]
        for sub in node.body:
            if isinstance(sub, _FuncDef):
                m_start, m_end = _start(sub), _end(sub)
                if m_end - m_start + 1 >= min_lines:
                    chunks.append(
                        self._mk(
                            text=_segment(lines, m_start, m_end),
                            file_name=file_name,
                            symbol_name=f"{node.name}.{sub.name}",
                            symbol_type="method",
                            start=m_start,
                            end=m_end,
                            parent_id=class_id,
                        )
                    )
        return chunks

    def scan(self, file_path: Path, config: ProjectConfig) -> list[IndexEntry]:
        file_path = Path(file_path)
        file_name = file_path.name
        source = file_path.read_text(encoding="utf-8", errors="replace")
        if not source.strip():
            return []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return CodeChunker().scan(file_path, config)

        abs_path = str(file_path.resolve())
        entries: list[IndexEntry] = []
        for node in tree.body:
            if isinstance(node, _FuncDef):
                entries.append(
                    IndexEntry(
                        concept=node.name,
                        file_name=file_name,
                        file_path=abs_path,
                        start_line=node.lineno,
                        end_line=_end(node),
                        entry_type="function",
                    )
                )
            elif isinstance(node, ast.ClassDef):
                entries.append(
                    IndexEntry(
                        concept=node.name,
                        file_name=file_name,
                        file_path=abs_path,
                        start_line=node.lineno,
                        end_line=_end(node),
                        entry_type="class",
                    )
                )
                for sub in node.body:
                    if isinstance(sub, _FuncDef):
                        entries.append(
                            IndexEntry(
                                concept=f"{node.name}.{sub.name}",
                                file_name=file_name,
                                file_path=abs_path,
                                start_line=sub.lineno,
                                end_line=_end(sub),
                                entry_type="method",
                            )
                        )
        return entries
