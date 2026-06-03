"""Tests for the chunker registry — the OCP dispatch mechanism."""

from code_rag.chunkers import ChunkerRegistry, get_registry
from code_rag.chunkers.code import CodeChunker
from code_rag.chunkers.lean import LeanChunker
from code_rag.chunkers.markdown import MarkdownChunker
from code_rag.chunkers.plaintext import PlaintextChunker
from code_rag.chunkers.python import PythonChunker


def test_resolves_known_extensions():
    registry = get_registry()
    assert isinstance(registry.get(".md"), MarkdownChunker)
    assert isinstance(registry.get(".py"), PythonChunker)  # stdlib-ast, overrides CodeChunker
    assert isinstance(registry.get(".rs"), CodeChunker)
    assert isinstance(registry.get(".lean"), LeanChunker)


def test_unknown_extension_uses_fallback():
    registry = get_registry()
    assert isinstance(registry.get(".unheardof"), PlaintextChunker)


def test_no_fallback_returns_none():
    registry = ChunkerRegistry()
    assert registry.get(".py") is None
