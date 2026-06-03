"""End-to-end ingest pipeline tests using fake providers (no network/Docker)."""

import pytest

from code_rag.ingest import ingest
from code_rag.providers import InMemoryVectorStore


def _seed_corpus(tmp_path):
    (tmp_path / "doc.md").write_text("## Intro\n\nSome documentation prose here.\n")
    (tmp_path / "code.py").write_text(
        "def compute(x):\n    # a few lines so enrichment does not skip it\n"
        "    y = x + 1\n    return y\n"
    )


@pytest.mark.asyncio
async def test_full_pipeline_indexes_chunks(
    tmp_path, make_config, fake_embeddings, in_memory_store
):
    _seed_corpus(tmp_path)
    config = make_config(data_dir=tmp_path, file_extensions=[".md", ".py"])

    result = await ingest(config, embedder=fake_embeddings, store=in_memory_store)

    assert result["chunks"] > 0
    assert result["indexed"] == result["chunks"]
    assert result["pointer_entries"] > 0

    # The store is actually populated and searchable.
    query_vec = (await fake_embeddings.embed(["how to compute"]))[0]
    hits = in_memory_store.search(config.collection, query_vec, top_k=3)
    assert hits


@pytest.mark.asyncio
async def test_empty_data_dir_returns_zero_counts(
    tmp_path, make_config, fake_embeddings, in_memory_store
):
    config = make_config(data_dir=tmp_path, file_extensions=[".md", ".py"])

    result = await ingest(config, embedder=fake_embeddings, store=in_memory_store)

    assert result == {"chunks": 0, "indexed": 0, "pointer_entries": 0}


@pytest.mark.asyncio
async def test_enrichment_skipped_without_llm(
    tmp_path, make_config, fake_embeddings, in_memory_store
):
    _seed_corpus(tmp_path)
    config = make_config(data_dir=tmp_path, file_extensions=[".py"])

    await ingest(config, embedder=fake_embeddings, store=in_memory_store, llm=None)

    query_vec = (await fake_embeddings.embed(["compute"]))[0]
    hits = in_memory_store.search(config.collection, query_vec, top_k=5)
    assert hits
    assert all(h["summary"] is None for h in hits)


@pytest.mark.asyncio
async def test_enrichment_runs_with_llm(
    tmp_path, make_config, fake_embeddings, in_memory_store, fake_llm
):
    _seed_corpus(tmp_path)
    config = make_config(data_dir=tmp_path, file_extensions=[".py"])

    await ingest(config, embedder=fake_embeddings, store=in_memory_store, llm=fake_llm)

    assert len(fake_llm.calls) > 0
    query_vec = (await fake_embeddings.embed(["compute"]))[0]
    hits = in_memory_store.search(config.collection, query_vec, top_k=5)
    assert any(h["summary"] for h in hits)


# ---------------------------------------------------------------------------
# Chunk identity: same basename in different directories must not collide
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_basename_in_different_dirs_do_not_collide(
    tmp_path, make_config, fake_embeddings, in_memory_store
):
    # Two structurally identical files sharing a basename — the classic collision.
    for sub in ("alpha", "beta"):
        d = tmp_path / sub
        d.mkdir()
        (d / "utils.py").write_text(f"def helper():\n    # module in {sub}\n    return '{sub}'\n")
    config = make_config(data_dir=tmp_path, file_extensions=[".py"])

    result = await ingest(config, embedder=fake_embeddings, store=in_memory_store)

    # Both files survive: distinct source_paths, distinct points, nothing overwritten.
    stored = in_memory_store.list_source_paths(config.collection)
    assert stored == {"alpha/utils.py", "beta/utils.py"}
    assert result["indexed"] == result["chunks"]


# ---------------------------------------------------------------------------
# Re-ingest sync (--no-recreate): edits, deletions, idempotence
# ---------------------------------------------------------------------------


async def _reingest(config, embedder, store):
    return await ingest(config, embedder, store, recreate_collection=False)


@pytest.mark.asyncio
async def test_reingest_unchanged_is_idempotent(
    tmp_path, make_config, fake_embeddings, in_memory_store
):
    _seed_corpus(tmp_path)
    config = make_config(data_dir=tmp_path, file_extensions=[".md", ".py"])

    first = await _reingest(config, fake_embeddings, in_memory_store)
    count_after_first = len(in_memory_store._points)
    second = await _reingest(config, fake_embeddings, in_memory_store)

    assert first["chunks"] == second["chunks"]
    assert len(in_memory_store._points) == count_after_first


@pytest.mark.asyncio
async def test_reingest_after_edit_leaves_no_orphans(
    tmp_path, make_config, fake_embeddings, in_memory_store
):
    target = tmp_path / "code.py"
    target.write_text("def compute(x):\n    y = x + 1\n    return y\n")
    config = make_config(data_dir=tmp_path, file_extensions=[".py"])

    await _reingest(config, fake_embeddings, in_memory_store)

    # Prepend lines so every chunk's line range shifts — old IDs would orphan.
    target.write_text(
        "# new banner comment\n# pushing everything down\n\n"
        "def compute(x):\n    y = x + 1\n    return y\n"
    )
    await _reingest(config, fake_embeddings, in_memory_store)

    # A clean ingest into a fresh store is the ground truth for "current corpus".
    # The re-ingested store must match it exactly — no stale points from old ranges.
    fresh = await ingest(config, fake_embeddings, InMemoryVectorStore())
    assert len(in_memory_store._points) == fresh["chunks"]


@pytest.mark.asyncio
async def test_reingest_after_delete_prunes_file(
    tmp_path, make_config, fake_embeddings, in_memory_store
):
    (tmp_path / "keep.py").write_text("def keep():\n    return 1\n")
    gone = tmp_path / "gone.py"
    gone.write_text("def gone():\n    return 2\n")
    config = make_config(data_dir=tmp_path, file_extensions=[".py"])

    await _reingest(config, fake_embeddings, in_memory_store)
    assert "gone.py" in in_memory_store.list_source_paths(config.collection)

    gone.unlink()
    await _reingest(config, fake_embeddings, in_memory_store)

    stored = in_memory_store.list_source_paths(config.collection)
    assert stored == {"keep.py"}
