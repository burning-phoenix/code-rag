"""
Core data models shared across the pipeline.

These are deliberately flat dataclasses with ``None`` defaults: every downstream
consumer (chunkers, enrichment, indexer, server) handles all chunk types
uniformly, so a subclass per content type would add complexity without benefit.
"""

import hashlib
import uuid
from dataclasses import dataclass


@dataclass
class Chunk:
    """A logical section of a document (markdown, code, or plaintext)."""

    text: str
    file_name: str
    section_title: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    math_density: float = 0.0  # ratio of math-containing lines to total lines
    # Corpus-relative path (POSIX), stamped by chunk_directory. Used for point
    # identity and re-ingest sync; distinct from file_name, which is for display.
    source_path: str | None = None
    # --- Code support fields ---
    chunk_id: str | None = None  # stable UUID, assigned at creation (for parent-child links)
    chunk_type: str = "markdown"  # "markdown" | "code" | "plaintext"
    language: str | None = None  # programming language identifier (e.g. "python", "glsl")
    symbol_name: str | None = None  # function/class/method name
    symbol_type: str | None = None  # "function" | "class" | "method" | "module_preamble"
    parent_id: str | None = None  # chunk_id of parent chunk (class → method link)
    # --- Enrichment fields (populated optionally at ingest time) ---
    summary: str | None = None  # LLM-generated summary
    hypothetical_questions: list[str] | None = None  # LLM-generated questions this chunk answers


@dataclass
class IndexEntry:
    """A pointer from a concept to its location in a document."""

    concept: str
    file_name: str
    file_path: str
    start_line: int
    end_line: int
    entry_type: str  # "section", "definition", "theorem", ..., "class", "function", "method"


def point_id(chunk: Chunk) -> str:
    """Derive a vector-store point ID from a chunk's identity.

    The single source of truth for chunk identity — every store derives point
    IDs through this function so the rule cannot drift between implementations.
    Identity keys on ``source_path`` (the corpus-relative path) so two files
    sharing a basename in different directories never collide; it falls back to
    ``file_name`` only for chunks not stamped by ``chunk_directory``.
    """
    key = chunk.source_path or chunk.file_name
    ident = f"{key}:{chunk.chunk_type}:{chunk.start_line}-{chunk.end_line}"
    return str(uuid.UUID(hashlib.md5(ident.encode()).hexdigest()))
