"""Behaviour tests for the AST code chunker."""

from pathlib import Path

import pytest

from code_rag.chunkers.code import CodeChunker, _get_parser, _node_name

CHUNKER = CodeChunker()


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_two_top_level_functions(tmp_path, make_config):
    content = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    path = _write(tmp_path, "f.py", content)
    config = make_config(code_chunk_min_lines=1, code_chunk_max_lines=150)

    chunks = CHUNKER.chunk(path, config)

    funcs = [c for c in chunks if c.symbol_type == "function"]
    assert {c.symbol_name for c in funcs} == {"alpha", "beta"}


def test_large_class_splits_into_parent_and_method_children(tmp_path, make_config):
    content = (
        "class C:\n"
        "    def a(self):\n        return 1\n"
        "    def b(self):\n        return 2\n"
        "    def c(self):\n        return 3\n"
    )
    path = _write(tmp_path, "c.py", content)
    # Force the split: class exceeds max.
    config = make_config(code_chunk_min_lines=1, code_chunk_max_lines=3)

    chunks = CHUNKER.chunk(path, config)

    parents = [c for c in chunks if c.symbol_type == "class"]
    methods = [c for c in chunks if c.symbol_type == "method"]
    assert len(parents) == 1
    assert len(methods) == 3
    assert all(m.parent_id == parents[0].chunk_id for m in methods)


def test_tiny_method_is_not_merged_into_class_parent(tmp_path, make_config):
    """A short method must survive as its own child, not fold into the parent.

    Regression: the merge pass once absorbed any sub-min-lines chunk into the
    previous one, corrupting the class parent (duplicated text, wrong range) and
    erasing the method as a retrievable unit.
    """
    content = (
        "class C:\n"
        "    def tiny(self): return 1\n"  # 1-line method, under min_lines
        "    def b(self):\n        return 2\n"
        "    def c(self):\n        return 3\n"
    )
    path = _write(tmp_path, "c.py", content)
    # min_lines=5 makes the 1-line method "tiny"; max=3 forces the parent split.
    config = make_config(code_chunk_min_lines=5, code_chunk_max_lines=3)

    chunks = CHUNKER.chunk(path, config)

    parents = [c for c in chunks if c.symbol_type == "class"]
    methods = [c for c in chunks if c.symbol_type == "method"]
    assert len(parents) == 1
    # All three methods preserved, links intact, none merged into the parent.
    assert {m.symbol_name for m in methods} == {"C.tiny", "C.b", "C.c"}
    assert all(m.parent_id == parents[0].chunk_id for m in methods)


def test_small_class_single_chunk(tmp_path, make_config):
    content = "class Tiny:\n    def a(self):\n        return 1\n"
    path = _write(tmp_path, "t.py", content)
    config = make_config(code_chunk_min_lines=1, code_chunk_max_lines=150)

    chunks = CHUNKER.chunk(path, config)

    assert len(chunks) == 1
    assert chunks[0].symbol_type == "class"
    assert chunks[0].symbol_name == "Tiny"


def test_module_preamble_chunk(tmp_path, make_config):
    content = "import os\n\nCONST = 1\n\n\ndef work():\n    return CONST\n"
    path = _write(tmp_path, "m.py", content)
    config = make_config(code_chunk_min_lines=1, code_chunk_max_lines=150)

    chunks = CHUNKER.chunk(path, config)

    assert any(c.symbol_type == "module_preamble" for c in chunks)


def test_empty_file_returns_no_chunks(tmp_path, make_config):
    path = _write(tmp_path, "empty.py", "")
    assert CHUNKER.chunk(path, make_config()) == []


@pytest.mark.parametrize(
    "language,source,expected",
    [
        ("python", "def foo():\n    pass\n", "foo"),
        ("rust", "impl Widget {\n    fn bar(&self) {}\n}\n", "Widget"),
        ("c", "int main(void) {\n    return 0;\n}\n", "main"),
    ],
)
def test_node_name_extraction(language, source, expected):
    parser = _get_parser(language)
    root = parser.parse(source.encode()).root_node
    node = root.children[0]
    assert _node_name(node, language) == expected
