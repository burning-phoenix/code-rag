"""Tests for project configuration loading."""

import json

import pytest

from code_rag.config import load_config, reset_config

VALID_CONFIG = {
    "name": "demo-rag",
    "collection": "demo",
    "data_dir": "RAG_data",
    "pointer_index": "pointer_index.json",
    "description": "demo project",
    "key_terms": ["alpha", "beta"],
    "qdrant_url": "http://file-url:6333",
    "file_extensions": [".py", ".md"],
    "enrich_chunks": True,
}


def _write_config(tmp_path, data=VALID_CONFIG):
    (tmp_path / "rag-config.json").write_text(json.dumps(data))


def test_valid_config_loads(tmp_path):
    _write_config(tmp_path)
    config = load_config(tmp_path)

    assert config.name == "demo-rag"
    assert config.collection == "demo"
    assert config.key_terms == ["alpha", "beta"]
    assert config.file_extensions == [".py", ".md"]
    assert config.enrich_chunks is True
    # Relative paths resolve against the project directory.
    assert config.data_dir == tmp_path / "RAG_data"
    assert config.pointer_index == tmp_path / "pointer_index.json"


def test_missing_config_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="rag-config.json"):
        load_config(tmp_path)


def test_env_vars_override_file(tmp_path, monkeypatch):
    _write_config(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://env-url:6333")
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key-123")

    config = load_config(tmp_path)

    assert config.qdrant_url == "http://env-url:6333"  # env beats file value
    assert config.openrouter_api_key == "env-key-123"


def test_reset_config_clears_cache(tmp_path, monkeypatch):
    _write_config(tmp_path)
    monkeypatch.setenv("RAG_PROJECT_DIR", str(tmp_path))

    first = load_config()
    assert load_config() is first  # cached on repeat call

    reset_config()
    assert load_config() is not first  # rebuilt after reset
