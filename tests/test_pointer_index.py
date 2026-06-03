"""Tests for the pointer index: scanning, search, and dispatch."""

from code_rag.chunkers import get_registry
from code_rag.pointer_index import build_index, search_index


def _entry(concept, entry_type="section", file_name="f.md"):
    return {
        "concept": concept,
        "file_name": file_name,
        "file_path": f"/abs/{file_name}",
        "start_line": 1,
        "end_line": 10,
        "entry_type": entry_type,
    }


def test_search_exact_match():
    index = [_entry("Backpropagation")]
    results = search_index(index, "Backpropagation")
    assert len(results) == 1
    assert results[0]["concept"] == "Backpropagation"


def test_search_is_bidirectional():
    index = [_entry("Scene", entry_type="class"), _entry("Scene.play", entry_type="method")]
    # Query contained in concept ("Scene" -> "Scene.play") and concept contained
    # in query ("Scene" <- "Scene.play") both match.
    assert len(search_index(index, "Scene")) == 2
    assert len(search_index(index, "Scene.play")) == 2


def test_search_no_match():
    assert search_index([_entry("Alpha")], "nonexistent") == []


def test_search_sorts_exact_before_partial_then_by_type():
    index = [
        _entry("Scene.play", entry_type="method"),  # partial match for "scene"
        _entry("Scene", entry_type="class"),  # exact match
    ]
    results = search_index(index, "Scene")
    # Exact match first regardless of type priority.
    assert results[0]["concept"] == "Scene"
    assert results[1]["concept"] == "Scene.play"


def test_markdown_scan_extracts_headers_and_formal_objects(tmp_path, make_config):
    content = (
        "# Title\n\n## Background\n\n**Definition 2.1** A widget is a thing.\n\n## Conclusion\n"
    )
    path = tmp_path / "doc.md"
    path.write_text(content)
    registry = get_registry()

    entries = registry.get(".md").scan(path, make_config())

    concepts = {e["concept"] if isinstance(e, dict) else e.concept for e in entries}
    assert "Background" in concepts
    assert "Definition 2.1" in concepts


def test_code_scan_extracts_functions_and_classes(tmp_path, make_config):
    path = tmp_path / "m.py"
    path.write_text("def helper():\n    return 1\n\n\nclass Thing:\n    pass\n")
    registry = get_registry()

    entries = registry.get(".py").scan(path, make_config())

    concepts = {e.concept for e in entries}
    assert "helper" in concepts
    assert "Thing" in concepts


def test_build_index_routes_by_extension(tmp_path, make_config):
    (tmp_path / "doc.md").write_text("## Section A\n\nbody\n")
    (tmp_path / "code.py").write_text("def fn():\n    return 1\n")
    config = make_config(data_dir=tmp_path, file_extensions=[".md", ".py"])

    index = build_index(tmp_path, config, get_registry())

    types = {e["entry_type"] for e in index}
    concepts = {e["concept"] for e in index}
    assert "section" in types  # from the markdown scanner
    assert "function" in types  # from the code scanner
    assert {"Section A", "fn"} <= concepts
