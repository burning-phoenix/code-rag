"""Behaviour tests for the plaintext fallback chunker."""

from pathlib import Path

from code_rag.chunkers.plaintext import PlaintextChunker

CHUNKER = PlaintextChunker()

FOUR_PARAGRAPHS = "para one\n\npara two\n\npara three\n\npara four\n"


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "doc.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_paragraphs_become_separate_chunks(tmp_path, make_config):
    path = _write(tmp_path, FOUR_PARAGRAPHS)
    # max=1 flushes at every blank-line boundary; min=0 disables merging.
    config = make_config(min_chunk_lines=0, max_chunk_lines=1)

    chunks = CHUNKER.chunk(path, config)

    assert len(chunks) == 4
    assert all(c.chunk_type == "plaintext" for c in chunks)


def test_small_paragraphs_merge(tmp_path, make_config):
    path = _write(tmp_path, FOUR_PARAGRAPHS)
    config = make_config(min_chunk_lines=1000, max_chunk_lines=1)

    chunks = CHUNKER.chunk(path, config)

    assert len(chunks) < 4


def test_empty_file_returns_no_chunks(tmp_path, make_config):
    path = _write(tmp_path, "")
    assert CHUNKER.chunk(path, make_config()) == []
