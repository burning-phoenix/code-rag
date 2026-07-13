"""Tests for the eval report builders (pure formatting over QueryResults)."""

from code_rag.eval.dataset import Query, Span
from code_rag.eval.metrics import QueryResult
from code_rag.eval.report import (
    CANARY_LABEL,
    build_concentration_table,
    build_coverage_table,
    build_geometry_table,
    build_markdown_report,
    build_wholeness_table,
    canary_check,
    chunk_geometry,
)
from code_rag.models import Chunk

WEIGHTS = {"a.py": [2] * 100}

QUERY = Query(
    id="q1",
    query="x",
    tool="search_code",
    decisive_spans=[Span(file="pkg/a.py", symbol="foo", start_line=10, end_line=20)],
)
HIT = {
    "source_path": "pkg/a.py",
    "file_name": "a.py",
    "start_line": 12,
    "end_line": 18,
}
RESULTS = {"ast+enrich": [QueryResult(query=QUERY, retrieved=[HIT])]}
TOP_K = [3, 5]


def test_chunk_geometry_math():
    chunks = [
        Chunk(text="", file_name="a.py", section_title="", start_line=1, end_line=10),
        Chunk(text="", file_name="a.py", section_title="", start_line=11, end_line=30),
    ]
    g = chunk_geometry(chunks)
    assert g["count"] == 2
    assert g["max_lines"] == 20  # second chunk is 20 lines
    assert g["mean_lines"] == 15.0


def test_chunk_geometry_empty():
    assert chunk_geometry([])["count"] == 0


def test_tables_render_as_markdown():
    cov = build_coverage_table(RESULTS, WEIGHTS, TOP_K)
    assert cov.startswith("| coverage |")
    assert "ast+enrich" in cov
    assert "| coverage@3 (decisive) |" in cov

    con = build_concentration_table(RESULTS, WEIGHTS, TOP_K)
    assert "| concentration@3 |" in con

    whole = build_wholeness_table(RESULTS, TOP_K)
    assert "overall wholeness@3" in whole
    assert "(n=" in whole  # wholeness denominators vary per cell

    geom = build_geometry_table({"ast+enrich": chunk_geometry([])})
    assert "| chunks |" in geom


def test_markdown_report_has_sections_and_definitions():
    geometry = {
        "ast+enrich": {"count": 5, "mean_lines": 12.0, "median_lines": 10.0, "max_lines": 40}
    }
    doc = build_markdown_report(RESULTS, geometry, TOP_K, "## Setup\n\nfixture.", WEIGHTS)
    for heading in (
        "# code-rag retrieval evaluation",
        "## Setup",
        "## Metrics",
        "## Results",
        "Coverage",
        "Concentration",
        "Wholeness",
        "Chunk geometry",
        "## Caveats",
    ):
        assert heading in doc
    # the definitions section explains the framing and the canary
    assert "did the answer arrive" in doc
    assert "canary" in doc


# --- canary alarm -------------------------------------------------------------

# The canary retrieves the span exactly (coverage 1, concentration 1); the real
# config retrieves nothing. A suite that lets this stand must flag itself.
CANARY_WINS = {
    "ast": [QueryResult(query=QUERY, retrieved=[])],
    CANARY_LABEL: [QueryResult(query=QUERY, retrieved=[HIT])],
}


def test_canary_alarm_fires_when_canary_wins_both_metrics():
    warnings = canary_check(CANARY_WINS, WEIGHTS, TOP_K)
    assert warnings
    doc = build_markdown_report(CANARY_WINS, {}, TOP_K, "## Setup", WEIGHTS)
    assert "## Warning: metric self-check failed" in doc


def test_canary_alarm_silent_when_a_real_config_wins():
    whole_file = {"source_path": "pkg/a.py", "file_name": "a.py", "start_line": 1, "end_line": 100}
    normal = {
        "ast": [QueryResult(query=QUERY, retrieved=[HIT])],
        CANARY_LABEL: [QueryResult(query=QUERY, retrieved=[whole_file])],
    }
    assert canary_check(normal, WEIGHTS, TOP_K) == []
    doc = build_markdown_report(normal, {}, TOP_K, "## Setup", WEIGHTS)
    assert "## Canary alarm" not in doc


def test_canary_check_noop_without_canary_column():
    assert canary_check(RESULTS, WEIGHTS, TOP_K) == []
