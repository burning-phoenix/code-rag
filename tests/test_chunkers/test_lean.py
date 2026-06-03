"""Behaviour tests for the Lean chunker (regex-based, comment-aware)."""

from code_rag.chunkers.lean import LeanChunker

CHUNKER = LeanChunker()


def test_supported_extensions():
    assert CHUNKER.supported_extensions() == {".lean"}


def test_declarations_get_correct_symbol_types(fixtures_dir, make_config):
    chunks = CHUNKER.chunk(fixtures_dir / "sample.lean", make_config())

    by_name = {c.symbol_name: c.symbol_type for c in chunks}
    assert by_name["double"] == "function"
    assert by_name["double_eq"] == "theorem"
    assert by_name["double_zero"] == "lemma"


def test_preamble_is_captured(fixtures_dir, make_config):
    chunks = CHUNKER.chunk(fixtures_dir / "sample.lean", make_config())

    preambles = [c for c in chunks if c.symbol_type == "module_preamble"]
    assert len(preambles) == 1
    assert "import" in preambles[0].text


def test_keywords_inside_comments_are_not_declarations(fixtures_dir, make_config):
    chunks = CHUNKER.chunk(fixtures_dir / "sample.lean", make_config())

    # Only the three real declarations — comment mentions are ignored.
    decl_names = {c.symbol_name for c in chunks if c.symbol_type != "module_preamble"}
    assert decl_names == {"double", "double_eq", "double_zero"}


def test_scan_maps_entry_types(fixtures_dir, make_config):
    entries = CHUNKER.scan(fixtures_dir / "sample.lean", make_config())

    by_concept = {e.concept: e.entry_type for e in entries}
    assert by_concept["double"] == "definition"
    assert by_concept["double_eq"] == "theorem"
    assert by_concept["double_zero"] == "lemma"


def _write_lean(tmp_path, content):
    path = tmp_path / "d.lean"
    path.write_text(content, encoding="utf-8")
    return path


def test_doc_comment_attaches_to_its_own_declaration(tmp_path, make_config):
    content = (
        "import Foo\n\n"
        "/-- The first theorem. -/\n"
        "theorem first : True := trivial\n\n"
        "/-- The second theorem. -/\n"
        "theorem second : True := trivial\n"
    )
    chunks = CHUNKER.chunk(_write_lean(tmp_path, content), make_config())
    by_name = {c.symbol_name: c.text for c in chunks}

    # Each docstring travels with the declaration it documents...
    assert "The first theorem." in by_name["first"]
    assert "The second theorem." in by_name["second"]
    # ...and does NOT leak into the previous declaration's chunk.
    assert "The second theorem." not in by_name["first"]


def test_multiline_doc_comment_is_fully_absorbed(tmp_path, make_config):
    content = "/-- line one\n    line two -/\ntheorem ml : True := trivial\n"
    chunks = CHUNKER.chunk(_write_lean(tmp_path, content), make_config())
    text = next(c.text for c in chunks if c.symbol_name == "ml")
    assert "line one" in text
    assert "line two" in text


def test_attribute_line_attaches_to_following_declaration(tmp_path, make_config):
    content = "import Foo\n\n@[simp]\ntheorem tagged : True := trivial\n"
    chunks = CHUNKER.chunk(_write_lean(tmp_path, content), make_config())
    text = next(c.text for c in chunks if c.symbol_name == "tagged")
    assert "@[simp]" in text


def test_preamble_excludes_first_declaration_doc(tmp_path, make_config):
    content = "import Foo\n\n/-- Documented. -/\ndef thing : Nat := 0\n"
    chunks = CHUNKER.chunk(_write_lean(tmp_path, content), make_config())
    preamble = next(c.text for c in chunks if c.symbol_type == "module_preamble")
    assert "import Foo" in preamble
    assert "Documented." not in preamble
