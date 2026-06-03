"""Behaviour tests for the stdlib-ast Python chunker."""

from pathlib import Path

from code_rag.chunkers.python import PythonChunker

CHUNKER = PythonChunker()


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "m.py"
    path.write_text(content, encoding="utf-8")
    return path


def test_class_yields_parent_plus_substantial_methods(tmp_path, make_config):
    content = (
        "class Widget:\n"
        "    def area(self):\n"  # 5 lines — substantial
        "        w = self.w\n"
        "        h = self.h\n"
        "        total = w * h\n"
        "        return total\n"
        "    def name(self): return 'w'\n"  # 1 line — trivial
    )
    path = _write(tmp_path, content)
    config = make_config(code_chunk_min_lines=5)

    chunks = CHUNKER.chunk(path, config)

    parents = [c for c in chunks if c.symbol_type == "class"]
    methods = [c for c in chunks if c.symbol_type == "method"]
    assert [p.symbol_name for p in parents] == ["Widget"]
    # The trivial one-liner stays inside the parent; only the substantial method splits.
    assert {m.symbol_name for m in methods} == {"Widget.area"}
    assert all(m.parent_id == parents[0].chunk_id for m in methods)


def test_decorators_are_included_in_chunk_span(tmp_path, make_config):
    content = "import functools\n\n\n@functools.cache\ndef memoized(n):\n    return n * n\n"
    path = _write(tmp_path, content)
    config = make_config()

    chunks = CHUNKER.chunk(path, config)
    fn = next(c for c in chunks if c.symbol_name == "memoized")
    assert "@functools.cache" in fn.text
    assert fn.start_line == 4  # the decorator line, not the def line


def test_method_decorator_included(tmp_path, make_config):
    content = (
        "class C:\n"
        "    @property\n"
        "    def value(self):\n"
        "        x = compute()\n"
        "        y = adjust(x)\n"
        "        return y\n"
    )
    path = _write(tmp_path, content)
    chunks = CHUNKER.chunk(path, make_config(code_chunk_min_lines=3))
    method = next(c for c in chunks if c.symbol_name == "C.value")
    assert "@property" in method.text


def test_module_preamble_captured(tmp_path, make_config):
    content = "import os\nimport sys\n\nCONST = 3\n\n\ndef work():\n    return CONST\n"
    path = _write(tmp_path, content)
    chunks = CHUNKER.chunk(path, make_config())
    assert any(c.symbol_type == "module_preamble" for c in chunks)
    assert any(c.symbol_name == "work" for c in chunks)


def test_nested_def_is_not_a_separate_chunk(tmp_path, make_config):
    content = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    path = _write(tmp_path, content)
    chunks = CHUNKER.chunk(path, make_config(code_chunk_min_lines=1))
    names = {c.symbol_name for c in chunks}
    assert "outer" in names
    assert "inner" not in names  # nested closure stays within the outer chunk


def test_large_top_level_function_is_split(tmp_path, make_config):
    body = "\n".join(f"    step_{i} = {i}" for i in range(20))
    content = f"def big():\n{body}\n    return step_0\n"
    path = _write(tmp_path, content)
    chunks = CHUNKER.chunk(path, make_config(code_chunk_max_lines=8))
    parts = [c for c in chunks if c.symbol_name == "big"]
    assert len(parts) >= 2
    # Disjoint line ranges so point identities don't collide.
    spans = sorted((c.start_line, c.end_line) for c in parts)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert next_start > prev_end


def test_syntax_error_falls_back_to_tree_sitter(tmp_path, make_config):
    # Python-2 print statement: ast.parse raises; tree-sitter still finds `good`.
    content = 'def good():\n    return 1\n\nprint "python2 syntax"\n'
    path = _write(tmp_path, content)
    chunks = CHUNKER.chunk(path, make_config())
    assert any(c.symbol_name == "good" for c in chunks)


def test_empty_file_returns_no_chunks(tmp_path, make_config):
    path = _write(tmp_path, "\n  \n")
    assert CHUNKER.chunk(path, make_config()) == []


def test_scan_indexes_functions_classes_methods(tmp_path, make_config):
    content = "def top():\n    return 1\n\nclass C:\n    def m(self):\n        return 2\n"
    path = _write(tmp_path, content)
    entries = CHUNKER.scan(path, make_config())
    by_concept = {e.concept: e.entry_type for e in entries}
    assert by_concept.get("top") == "function"
    assert by_concept.get("C") == "class"
    assert by_concept.get("C.m") == "method"
