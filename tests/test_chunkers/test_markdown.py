"""Behaviour tests for the markdown chunker."""

from pathlib import Path

from code_rag.chunkers.markdown import MarkdownChunker

CHUNKER = MarkdownChunker()


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_supported_extensions():
    assert CHUNKER.supported_extensions() == {".md"}


def test_three_headers_produce_three_chunks(tmp_path, make_config):
    content = "## Alpha\nalpha body\n\n## Beta\nbeta body\n\n## Gamma\ngamma body\n"
    path = _write(tmp_path, "doc.md", content)
    # Disable merge/split so the header split is observed directly.
    config = make_config(min_chunk_lines=0, max_chunk_lines=0)

    chunks = CHUNKER.chunk(path, config)

    assert [c.section_title for c in chunks] == ["Alpha", "Beta", "Gamma"]


def test_small_sections_merge(tmp_path, make_config):
    content = "## Alpha\na\n\n## Beta\nb\n\n## Gamma\nc\n"
    path = _write(tmp_path, "doc.md", content)
    config = make_config(min_chunk_lines=50, max_chunk_lines=0)

    chunks = CHUNKER.chunk(path, config)

    assert len(chunks) < 3


def test_large_section_splits_at_paragraph_boundaries(tmp_path, make_config):
    paragraphs = "\n\n".join(f"Paragraph {i} line of prose." for i in range(12))
    content = f"## Big\n{paragraphs}\n"
    path = _write(tmp_path, "doc.md", content)
    config = make_config(min_chunk_lines=0, max_chunk_lines=5)

    chunks = CHUNKER.chunk(path, config)

    assert len(chunks) > 1


def test_split_preserves_display_math_block(tmp_path, make_config):
    filler = "\n\n".join(f"Filler paragraph number {i}." for i in range(8))
    math_block = "$$\na = b + c\nd = e + f\n$$"
    content = f"## Math\n{filler}\n\n{math_block}\n\n{filler}\n"
    path = _write(tmp_path, "doc.md", content)
    config = make_config(min_chunk_lines=0, max_chunk_lines=4)

    chunks = CHUNKER.chunk(path, config)

    # The display block must survive intact inside exactly one chunk.
    containing = [c for c in chunks if "a = b + c" in c.text]
    assert len(containing) == 1
    assert containing[0].text.count("$$") == 2


def test_empty_file_returns_no_chunks(tmp_path, make_config):
    path = _write(tmp_path, "empty.md", "")
    assert CHUNKER.chunk(path, make_config()) == []


def test_no_headers_single_chunk_titled_by_filename(tmp_path, make_config):
    content = "Just some prose.\nNo headers at all here.\n"
    path = _write(tmp_path, "plain.md", content)
    config = make_config(min_chunk_lines=0, max_chunk_lines=0)

    chunks = CHUNKER.chunk(path, config)

    assert len(chunks) == 1
    assert chunks[0].section_title == "plain.md"


def test_math_density_is_computed(tmp_path, make_config):
    # 2 inline-math lines, 2 prose lines → density ~0.5
    content = "$a = 1$\nplain text line\n$b = 2$\nmore prose here\n"
    path = _write(tmp_path, "m.md", content)
    config = make_config(min_chunk_lines=0, max_chunk_lines=0)

    chunks = CHUNKER.chunk(path, config)

    assert len(chunks) == 1
    assert 0.4 <= chunks[0].math_density <= 0.6
