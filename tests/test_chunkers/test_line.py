"""Behaviour tests for the eval-only line chunker (baseline + whole-file canary)."""

from pathlib import Path

from code_rag.eval.line_chunker import (
    WHOLE_FILE_EMBED_CHARS,
    WHOLE_FILE_WINDOW,
    LineChunker,
)


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "big.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_fixed_windows_cover_the_file(tmp_path, make_config):
    path = _write(tmp_path, "\n".join(f"line {i}" for i in range(1, 101)) + "\n")
    chunks = LineChunker(window=40).chunk(path, make_config())
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 40), (41, 80), (81, 100)]


def test_whole_file_canary_truncates_text_but_keeps_full_line_range(tmp_path, make_config):
    """Regression for the 2026-07-09 crash: a whole-file chunk of a large
    document blew past the embedding model's input limit. The embedded text is
    capped; the line range (all the metrics read) stays whole-file."""
    n_lines = 2000
    path = _write(tmp_path, ("x" * 100 + "\n") * n_lines)  # ~200k chars
    chunker = LineChunker(WHOLE_FILE_WINDOW, max_chars=WHOLE_FILE_EMBED_CHARS)

    chunks = chunker.chunk(path, make_config())

    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, n_lines)
    assert len(chunks[0].text) == WHOLE_FILE_EMBED_CHARS


def test_no_cap_by_default(tmp_path, make_config):
    path = _write(tmp_path, ("y" * 100 + "\n") * 50)
    chunks = LineChunker(window=40).chunk(path, make_config())
    assert all(len(c.text) > 100 for c in chunks)  # untruncated
