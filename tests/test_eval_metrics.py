"""Unit tests for the (pure) eval metrics and dataset loader — no network."""

from pathlib import Path

from code_rag.eval.dataset import Query, Span, load_dataset
from code_rag.eval.metrics import (
    QueryResult,
    hit_rate,
    item_hits_span,
    line_iou,
    mean_iou_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

DECISIVE_SPAN = Span(file="a.py", symbol="foo", start_line=10, end_line=20)
SUPPORTIVE_SPAN = Span(file="a.py", symbol="Bar")

QUERY = Query(
    id="q1",
    query="x",
    tool="search_code",
    decisive_spans=[DECISIVE_SPAN],
    supportive_spans=[SUPPORTIVE_SPAN],
)

HIT_DECISIVE = {"file_name": "a.py", "start_line": 12, "end_line": 18, "symbol_name": "foo"}
HIT_SUPPORTIVE = {"file_name": "a.py", "start_line": 50, "end_line": 60, "symbol_name": "Bar"}
MISS = {"file_name": "b.py", "start_line": 1, "end_line": 5, "symbol_name": "baz"}


# --- item_hits_span ---------------------------------------------------------


def test_hit_by_line_overlap():
    assert item_hits_span(HIT_DECISIVE, DECISIVE_SPAN)


def test_hit_by_symbol_name():
    item = {"file_name": "a.py", "start_line": 999, "end_line": 1000, "symbol_name": "foo"}
    assert item_hits_span(item, DECISIVE_SPAN)


def test_no_hit_different_file():
    assert not item_hits_span(MISS, DECISIVE_SPAN)


def test_symbol_subset_match():
    item = {"file_name": "a.py", "symbol_name": "Bar.method", "start_line": 1, "end_line": 2}
    assert item_hits_span(item, SUPPORTIVE_SPAN)


def test_source_path_distinguishes_same_basename():
    """When the span names a path, matching keys on source_path, not basename."""
    span = Span(file="alpha/utils.py", symbol="helper")
    same = {"source_path": "alpha/utils.py", "file_name": "utils.py", "symbol_name": "helper"}
    other = {"source_path": "beta/utils.py", "file_name": "utils.py", "symbol_name": "helper"}
    assert item_hits_span(same, span)
    assert not item_hits_span(other, span)


def test_basename_fallback_without_source_path():
    """Bare-basename spans (or items lacking source_path) still match on basename."""
    span = Span(file="utils.py", symbol="helper")
    item = {"file_name": "utils.py", "symbol_name": "helper", "start_line": 1, "end_line": 2}
    assert item_hits_span(item, span)


def test_recall_span_filter_slices_by_content_type():
    from pathlib import Path

    from code_rag.eval.metrics import count_spans

    py_query = Query(
        id="qpy",
        query="x",
        tool="search_code",
        decisive_spans=[Span(file="a.py", symbol="foo", start_line=10, end_line=20)],
    )
    lean_query = Query(
        id="qlean",
        query="y",
        tool="search_documents",
        decisive_spans=[Span(file="b.lean", symbol="bar", start_line=1, end_line=5)],
    )
    py_hit = {"file_name": "a.py", "start_line": 12, "end_line": 18, "symbol_name": "foo"}
    results = [
        QueryResult(query=py_query, retrieved=[py_hit]),
        QueryResult(query=lean_query, retrieved=[]),  # lean span not retrieved
    ]

    def is_py(s: Span) -> bool:
        return Path(s.file).suffix == ".py"

    def is_lean(s: Span) -> bool:
        return Path(s.file).suffix == ".lean"

    assert count_spans(results, 2, is_py) == 1
    assert count_spans(results, 2, is_lean) == 1
    assert recall_at_k(results, 5, 2, is_py) == 1.0  # python span found
    assert recall_at_k(results, 5, 2, is_lean) == 0.0  # lean span missed
    assert recall_at_k(results, 5, 2) == 0.5  # unfiltered: 1 of 2


# --- line-range IoU ---------------------------------------------------------


def test_line_iou_tight_chunk_scores_high():
    span = Span(file="a.py", symbol="m", start_line=10, end_line=15)  # 6 lines
    tight = {"file_name": "a.py", "start_line": 10, "end_line": 15}  # exact
    assert line_iou(tight, span) == 1.0
    near = {"file_name": "a.py", "start_line": 10, "end_line": 16}  # 6/7
    assert abs(line_iou(near, span) - 6 / 7) < 1e-9


def test_line_iou_large_window_scores_low():
    span = Span(file="a.py", symbol="m", start_line=10, end_line=14)  # 5 lines
    window = {"file_name": "a.py", "start_line": 1, "end_line": 40}  # 40-line window
    # intersection 5, union 40 → 0.125
    assert abs(line_iou(window, span) - 0.125) < 1e-9


def test_line_iou_zero_for_disjoint_file_or_symbol_only():
    span = Span(file="a.py", symbol="m", start_line=10, end_line=14)
    assert line_iou({"file_name": "b.py", "start_line": 10, "end_line": 14}, span) == 0.0
    assert line_iou({"file_name": "a.py", "start_line": 50, "end_line": 60}, span) == 0.0
    symbol_only = Span(file="a.py", symbol="m")  # no line range
    assert line_iou({"file_name": "a.py", "start_line": 10, "end_line": 14}, symbol_only) == 0.0


def test_mean_iou_takes_best_item_per_span():
    query = Query(
        id="q",
        query="x",
        tool="search_code",
        decisive_spans=[Span(file="a.py", symbol="m", start_line=10, end_line=15)],
    )
    loose = {"file_name": "a.py", "start_line": 1, "end_line": 40}
    tight = {"file_name": "a.py", "start_line": 10, "end_line": 15}
    results = [QueryResult(query=query, retrieved=[loose, tight])]
    assert mean_iou_at_k(results, 5, 2) == 1.0  # best item (tight) wins
    # top-1 has only the loose 40-line window: inter 6 / union 40 = 0.15
    assert abs(mean_iou_at_k(results, 1, 2) - 0.15) < 1e-9


# --- ranked metrics ---------------------------------------------------------

# Decisive hit at rank 2, supportive hit at rank 3.
RESULTS = [QueryResult(query=QUERY, retrieved=[MISS, HIT_DECISIVE, HIT_SUPPORTIVE])]


def test_recall_respects_k_cutoff():
    assert recall_at_k(RESULTS, k=1, grade=2) == 0.0  # decisive not in top-1
    assert recall_at_k(RESULTS, k=2, grade=2) == 1.0  # decisive in top-2


def test_recall_supportive_grade():
    assert recall_at_k(RESULTS, k=3, grade=1) == 1.0


def test_hit_rate_decisive():
    assert hit_rate(RESULTS, grade=2) == 1.0


def test_mrr_reflects_rank():
    assert mrr_at_k(RESULTS, k=1) == 0.0  # decisive below the cutoff
    assert mrr_at_k(RESULTS, k=3) == 0.5  # decisive at rank 2 → 1/2


def test_ndcg_is_normalized_and_rank_sensitive():
    score = ndcg_at_k(RESULTS, k=3)
    assert 0.0 < score <= 1.0

    top_ranked = [QueryResult(query=QUERY, retrieved=[HIT_DECISIVE, HIT_SUPPORTIVE, MISS])]
    assert ndcg_at_k(top_ranked, k=3) > score


def test_ndcg_caps_at_one_with_overlapping_items():
    """Many items hitting the SAME single span must not push NDCG above 1.

    Regression: the line-window chunker emits overlapping chunks that each hit
    one decisive span; crediting every item inflated NDCG to >1.
    """
    one_span = Query(
        id="q1span",
        query="x",
        tool="search_code",
        decisive_spans=[Span(file="a.py", symbol="foo", start_line=10, end_line=20)],
    )
    # Three retrieved windows that all overlap the single gold span.
    windows = [
        {"file_name": "a.py", "start_line": 10, "end_line": 14, "symbol_name": "foo"},
        {"file_name": "a.py", "start_line": 13, "end_line": 17, "symbol_name": "foo"},
        {"file_name": "a.py", "start_line": 16, "end_line": 20, "symbol_name": "foo"},
    ]
    results = [QueryResult(query=one_span, retrieved=windows)]
    score = ndcg_at_k(results, k=5)
    assert score == 1.0  # span covered at rank 1; redundant windows add nothing


def test_empty_results_are_zero():
    assert recall_at_k([], k=5, grade=2) == 0.0
    assert hit_rate([], grade=2) == 0.0
    assert mrr_at_k([], k=5) == 0.0
    assert ndcg_at_k([], k=5) == 0.0


# --- dataset loader ---------------------------------------------------------


def test_loader_tolerates_empty_queries(tmp_path):
    # The loader must no-op cleanly on a placeholder dataset (empty queries).
    path = tmp_path / "ds.json"
    path.write_text('{"corpus_dir": "corpus", "queries": []}')
    dataset = load_dataset(path)
    assert dataset.corpus_dir == "corpus"
    assert dataset.queries == []


def test_real_golden_dataset_loads_and_validates():
    dataset_path = Path(__file__).parent / "eval" / "golden_dataset.json"
    dataset = load_dataset(dataset_path)
    assert dataset.corpus_dir == "corpus"
    assert len(dataset.queries) > 0
    # Every query has a valid tool and at least one decisive span.
    for q in dataset.queries:
        assert q.tool in {"search_code", "search_documents", "lookup_index"}
        assert q.decisive_spans
