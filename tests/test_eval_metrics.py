"""Unit tests for the (pure) eval metrics and dataset loader — no network.

The suite is three questions (coverage / concentration / wholeness); each test
group pins the cheat its metric exists to catch, plus the geometry-only hit
rule: symbols are never consulted, so a same-file item that doesn't overlap a
span scores nothing no matter what either side's symbol says.
"""

from pathlib import Path

from code_rag.eval.dataset import Query, Span, load_dataset
from code_rag.eval.metrics import (
    QueryResult,
    concentration_at_k,
    count_overlapped_spans,
    count_spans,
    coverage_at_k,
    load_corpus_weights,
    wholeness_at_k,
)

# a.py: 100 lines, 2 tokens each; b.lean: 50 lines, 3 tokens each.
WEIGHTS = {"a.py": [2] * 100, "b.lean": [3] * 50, "utils.py": [1] * 10}

DECISIVE_SPAN = Span(file="a.py", symbol="foo", start_line=10, end_line=20)  # 11 lines
SUPPORTIVE_SPAN = Span(file="a.py", symbol="Bar", start_line=50, end_line=60)  # 11 lines

QUERY = Query(
    id="q1",
    query="x",
    tool="search_code",
    decisive_spans=[DECISIVE_SPAN],
    supportive_spans=[SUPPORTIVE_SPAN],
)

EXACT = {"file_name": "a.py", "start_line": 10, "end_line": 20}
WHOLE_FILE = {"file_name": "a.py", "start_line": 1, "end_line": 100}
DISJOINT_SAME_FILE = {"file_name": "a.py", "start_line": 70, "end_line": 80, "symbol_name": None}


def one(retrieved: list[dict], query: Query = QUERY) -> list[QueryResult]:
    return [QueryResult(query=query, retrieved=retrieved)]


# --- coverage: did the answer arrive? ---------------------------------------


def test_coverage_full_for_containing_item():
    assert coverage_at_k(one([EXACT]), 5, WEIGHTS) == 1.0


def test_coverage_union_across_fragments_is_full():
    """Fragments' union counts — fair to chunkers that split the answer."""
    halves = [
        {"file_name": "a.py", "start_line": 10, "end_line": 14},
        {"file_name": "a.py", "start_line": 15, "end_line": 20},
    ]
    assert coverage_at_k(one(halves), 5, WEIGHTS) == 1.0


def test_coverage_grazing_scores_its_token_fraction_only():
    """The cheat binary recall allowed: graze one line, claim the span.

    Line 10 carries 5 tokens, lines 11–20 one each (span total 15); an item
    touching only line 10 covers 5/15 — token-weighted, not line-counted.
    """
    weights = {"a.py": [1] * 9 + [5] + [1] * 90}
    graze = {"file_name": "a.py", "start_line": 5, "end_line": 10}
    assert abs(coverage_at_k(one([graze]), 5, weights) - 5 / 15) < 1e-9


def test_coverage_respects_k_cutoff():
    results = one([DISJOINT_SAME_FILE, EXACT])
    assert coverage_at_k(results, 1, WEIGHTS) == 0.0
    assert coverage_at_k(results, 2, WEIGHTS) == 1.0


def test_coverage_supportive_grade():
    supp = {"file_name": "a.py", "start_line": 50, "end_line": 60}
    assert coverage_at_k(one([supp]), 5, WEIGHTS, grade=1) == 1.0
    assert coverage_at_k(one([supp]), 5, WEIGHTS, grade=2) == 0.0


def test_geometry_only_no_symbol_can_manufacture_a_hit():
    """Regression (in spirit) for the empty-symbol_name bug: a same-file item
    that doesn't overlap the span scores nothing on any metric, regardless of
    what symbol either side carries."""
    for item in (
        DISJOINT_SAME_FILE,
        {**DISJOINT_SAME_FILE, "symbol_name": "foo"},  # even a "matching" symbol
        {**DISJOINT_SAME_FILE, "symbol_name": ""},
    ):
        results = one([item])
        assert coverage_at_k(results, 5, WEIGHTS) == 0.0
        assert wholeness_at_k(results, 5) == 0.0
        assert count_overlapped_spans(results, 5) == 0


def test_source_path_distinguishes_same_basename():
    """When the span names a path, matching keys on source_path, not basename."""
    span = Span(file="alpha/utils.py", start_line=1, end_line=10)
    q = Query(id="q", query="x", tool="search_code", decisive_spans=[span])
    same = {
        "source_path": "alpha/utils.py",
        "file_name": "utils.py",
        "start_line": 1,
        "end_line": 10,
    }
    other = {
        "source_path": "beta/utils.py",
        "file_name": "utils.py",
        "start_line": 1,
        "end_line": 10,
    }
    assert coverage_at_k(one([same], q), 5, WEIGHTS) == 1.0
    assert coverage_at_k(one([other], q), 5, WEIGHTS) == 0.0


def test_coverage_span_filter_slices_by_content_type():
    py_query = Query(
        id="qpy",
        query="x",
        tool="search_code",
        decisive_spans=[Span(file="a.py", start_line=10, end_line=20)],
    )
    lean_query = Query(
        id="qlean",
        query="y",
        tool="search_documents",
        decisive_spans=[Span(file="b.lean", start_line=1, end_line=5)],
    )
    results = [
        QueryResult(query=py_query, retrieved=[EXACT]),
        QueryResult(query=lean_query, retrieved=[]),  # lean span not retrieved
    ]

    def is_py(s: Span) -> bool:
        return Path(s.file).suffix == ".py"

    def is_lean(s: Span) -> bool:
        return Path(s.file).suffix == ".lean"

    assert count_spans(results, 2, is_py) == 1
    assert count_spans(results, 2, is_lean) == 1
    assert coverage_at_k(results, 5, WEIGHTS, 2, is_py) == 1.0
    assert coverage_at_k(results, 5, WEIGHTS, 2, is_lean) == 0.0
    # unfiltered micro-average is token-weighted: 22 of 22+15 span tokens covered
    assert abs(coverage_at_k(results, 5, WEIGHTS, 2) - 22 / 37) < 1e-9


# --- concentration: how much of what the agent read was answer? -------------


def test_concentration_whole_file_maxes_coverage_and_dies_here():
    """The bloat cheat: one whole-file chunk swallows every gold line."""
    results = one([WHOLE_FILE])
    assert coverage_at_k(results, 5, WEIGHTS) == 1.0
    # useful = decisive (11 lines) + supportive (11 lines) of 100 read
    assert abs(concentration_at_k(results, 5, WEIGHTS) - 22 / 100) < 1e-9


def test_concentration_counts_supportive_as_useful():
    """Supportive context is labeled useful — it is not noise."""
    supp = {"file_name": "a.py", "start_line": 50, "end_line": 60}
    assert concentration_at_k(one([supp]), 5, WEIGHTS) == 1.0


def test_concentration_pays_per_read_for_duplicates():
    """A useful line retrieved twice costs twice and pays twice; a noise item
    of equal size halves the fraction."""
    assert concentration_at_k(one([EXACT, EXACT]), 5, WEIGHTS) == 1.0
    noise = {"file_name": "a.py", "start_line": 70, "end_line": 80}
    assert abs(concentration_at_k(one([EXACT, noise]), 5, WEIGHTS) - 0.5) < 1e-9


def test_concentration_averages_over_queries():
    q2 = Query(
        id="q2",
        query="y",
        tool="search_code",
        decisive_spans=[Span(file="a.py", start_line=30, end_line=40)],
    )
    results = [
        QueryResult(query=QUERY, retrieved=[EXACT]),  # 1.0
        QueryResult(query=q2, retrieved=[WHOLE_FILE]),  # 11/100
    ]
    assert abs(concentration_at_k(results, 5, WEIGHTS) - (1.0 + 11 / 100) / 2) < 1e-9


# --- wholeness: did each covered answer arrive in one piece? ----------------


def test_wholeness_single_containing_chunk_is_one():
    assert wholeness_at_k(one([EXACT]), 5) == 1.0
    padded = {"file_name": "a.py", "start_line": 8, "end_line": 25}
    assert wholeness_at_k(one([padded]), 5) == 1.0  # contained, even if padded


def test_wholeness_fragments_score_zero_while_coverage_is_full():
    """The failure coverage and concentration jointly miss: two tight fragments
    cover everything, yet no single piece can be trusted alone."""
    halves = [
        {"file_name": "a.py", "start_line": 10, "end_line": 14},
        {"file_name": "a.py", "start_line": 15, "end_line": 20},
    ]
    assert coverage_at_k(one(halves), 5, WEIGHTS) == 1.0
    assert wholeness_at_k(one(halves), 5) == 0.0


def test_wholeness_is_conditional_on_overlap():
    """Retrieve-nothing must not score a spurious 1.0; missed spans leave the
    denominator rather than counting as whole."""
    assert wholeness_at_k(one([]), 5) == 0.0
    assert count_overlapped_spans(one([]), 5) == 0
    two_spans = Query(
        id="q2s",
        query="x",
        tool="search_code",
        decisive_spans=[DECISIVE_SPAN, Span(file="a.py", start_line=30, end_line=40)],
    )
    results = one([EXACT], two_spans)  # first span whole, second untouched
    assert wholeness_at_k(results, 5) == 1.0
    assert count_overlapped_spans(results, 5) == 1


def test_empty_results_are_zero():
    assert coverage_at_k([], 5, WEIGHTS) == 0.0
    assert concentration_at_k([], 5, WEIGHTS) == 0.0
    assert wholeness_at_k([], 5) == 0.0


# --- corpus weights ----------------------------------------------------------


def test_load_corpus_weights_counts_whitespace_tokens_per_line(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.py").write_text("def foo():\n\n    return 1 + 2\n")
    weights = load_corpus_weights(tmp_path)
    assert weights["f.py"] == [2, 0, 4]  # keyed by basename, blank line = 0


# --- dataset loader ----------------------------------------------------------


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
    for q in dataset.queries:
        assert q.tool in {"search_code", "search_documents", "lookup_index"}
        assert q.decisive_spans
        # Labeling rule: geometry-only scoring means every decisive span must
        # carry a line range — a rangeless span is unscorable dead weight.
        for span in q.decisive_spans:
            assert span.start_line is not None and span.end_line is not None, (
                f"{q.id}: decisive span {span.file}::{span.symbol} has no line range"
            )
