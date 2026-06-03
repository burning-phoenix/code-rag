"""
Snapshot tests pinning chunker output so refactors that change chunk boundaries
are caught. Random ``chunk_id``/``parent_id`` UUIDs are normalised to stable
order-based tokens (preserving parent→child links) so snapshots are reproducible.

Regenerate after an intentional change with:
    pytest tests/test_snapshots.py --update-snapshots
"""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from code_rag.chunkers import get_registry

EXPECTED_DIR = Path(__file__).parent / "fixtures" / "expected"
CASES = ["sample.md", "sample.py", "sample.lean", "sample.txt"]


def _normalize(chunks) -> list[dict]:
    """Serialize chunks, replacing random UUIDs with stable tokens by order."""
    id_map: dict[str, str] = {}

    def token(cid: str | None) -> str | None:
        if cid is None:
            return None
        if cid not in id_map:
            id_map[cid] = f"chunk-{len(id_map)}"
        return id_map[cid]

    out = []
    for chunk in chunks:
        d = asdict(chunk)
        d["chunk_id"] = token(d["chunk_id"])
        d["parent_id"] = token(d["parent_id"])
        out.append(d)
    return out


@pytest.mark.parametrize("filename", CASES)
def test_chunk_snapshot(filename, fixtures_dir, make_config, update_snapshots):
    config = make_config(data_dir=fixtures_dir)
    chunker = get_registry().get(Path(filename).suffix.lower())
    chunks = chunker.chunk(fixtures_dir / filename, config)
    actual = _normalize(chunks)

    expected_path = EXPECTED_DIR / f"{filename.replace('.', '_')}_chunks.json"

    if update_snapshots:
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n")
        pytest.skip(f"updated snapshot: {expected_path.name}")

    assert expected_path.exists(), f"missing snapshot {expected_path.name}; run --update-snapshots"
    expected = json.loads(expected_path.read_text())
    assert actual == expected
