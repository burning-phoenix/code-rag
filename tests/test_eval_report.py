"""Tests for the eval report builders (pure formatting over QueryResults)."""

from code_rag.eval.dataset import Query, Span
from code_rag.eval.metrics import QueryResult
from code_rag.eval.report import (
    build_geometry_table,
    build_iou_table,
    build_markdown_report,
    build_report,
    chunk_geometry,
)
from code_rag.models import Chunk

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
    "symbol_name": "foo",
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
    report = build_report(RESULTS, TOP_K)
    assert report.startswith("| metric |")
    assert "ast+enrich" in report
    assert "| recall@3 (decisive) |" in report

    iou = build_iou_table(RESULTS, TOP_K)
    assert "overall IoU@3" in iou

    geom = build_geometry_table({"ast+enrich": chunk_geometry([])})
    assert "| chunks |" in geom


def test_markdown_report_has_sections_and_definitions():
    geometry = {
        "ast+enrich": {"count": 5, "mean_lines": 12.0, "median_lines": 10.0, "max_lines": 40}
    }
    doc = build_markdown_report(RESULTS, geometry, TOP_K, setup_md="## Setup\n\nfixture.")
    for heading in (
        "# code-rag retrieval evaluation",
        "## Setup",
        "## Metrics",
        "## Results",
        "Chunk geometry",
        "line-range IoU",
        "## Caveats",
    ):
        assert heading in doc
    # the definitions section explains what IoU measures
    assert "localization precision" in doc
