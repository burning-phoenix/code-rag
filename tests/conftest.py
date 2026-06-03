"""Shared test fixtures: fake providers, a config factory, and cache hygiene."""

from collections.abc import Callable
from pathlib import Path

import pytest

from code_rag.config import ProjectConfig, reset_config
from code_rag.models import Chunk
from code_rag.providers import FakeEmbeddings, FakeLLM, InMemoryVectorStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate chunker snapshot expectation files instead of asserting.",
    )


@pytest.fixture
def update_snapshots(request) -> bool:
    return bool(request.config.getoption("--update-snapshots"))


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Ensure the module-level config cache never leaks between tests."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., ProjectConfig]:
    """Factory for ProjectConfig with sensible test defaults; override via kwargs."""

    def _make(**overrides) -> ProjectConfig:
        defaults: dict = dict(
            name="test",
            collection="test",
            data_dir=tmp_path,
            pointer_index=tmp_path / "pointer_index.json",
            description="test project",
            key_terms=[],
            qdrant_url="",
            openrouter_api_key="",
            min_chunk_lines=20,
            max_chunk_lines=100,
            file_extensions=[".md", ".py", ".lean", ".txt"],
            enrich_chunks=False,
            enrichment_model="test-model",
            code_chunk_min_lines=5,
            code_chunk_max_lines=150,
        )
        defaults.update(overrides)
        return ProjectConfig(**defaults)

    return _make


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    # Small dimension keeps brute-force cosine fast in tests.
    return FakeEmbeddings(dimension=64)


@pytest.fixture
def in_memory_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        text="def foo():\n    return 42\n",
        file_name="sample.py",
        section_title="foo",
        start_line=1,
        end_line=2,
        chunk_type="code",
        language="python",
        symbol_name="foo",
        symbol_type="function",
    )
