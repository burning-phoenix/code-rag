"""
Retrieval metrics over hand-labelled, graded relevance spans.

The intended consumer of retrieval is an LLM agent: it reads the whole top-k
result set at once, pays (in context window and API cost) for every token, and
must trust each result without seeing the file around it. Retrieval can fail
that consumer in three distinct, measurable ways, so there are three metrics,
each computed at every k:

- **coverage** — did the answer's tokens arrive in the top-k results?
- **concentration** — what fraction of all retrieved tokens was answer?
- **wholeness** — did each answer that arrived arrive inside a single result?

A retrieved item (a chunk payload or pointer entry, normalised to a dict with
``file_name``/``start_line``/``end_line``) matches a gold span only when both
point at the same file and their line ranges overlap. Symbol names are never
used in matching: only the AST chunkers record them, so scoring through symbol
names would favour those chunkers over strategies that have no symbols to
report.

Scores are weighted by whitespace-separated tokens per line, not by line
count: a markdown "line" in this corpus is often a whole paragraph while a
Python line is around 40 characters, so line counts are not comparable across
the corpus's file types.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .dataset import Query, Span

DECISIVE = 2
SUPPORTIVE = 1

# basename -> whitespace-token count per line (index 0 = line 1)
LineWeights = Mapping[str, list[int]]


@dataclass
class QueryResult:
    """A query paired with the ranked items retrieved for it."""

    query: Query
    retrieved: list[dict]  # rank-ordered; each has file_name/start_line/end_line


def load_corpus_weights(corpus_dir: str | Path) -> dict[str, list[int]]:
    """Per-line whitespace-token counts for every corpus file, keyed by basename.

    Basename keying mirrors ``_same_file``'s fallback; the eval corpus keeps
    basenames unique, so no collision handling is attempted.
    """
    weights: dict[str, list[int]] = {}
    for p in sorted(Path(corpus_dir).rglob("*")):
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            weights[p.name] = [len(line.split()) for line in text.splitlines()]
    return weights


def _same_file(item: dict, span: Span) -> bool:
    """Match a retrieved item to a span's file.

    Prefers the full corpus-relative path when the item carries a ``source_path``
    and the span names a path (so same-basename files in different directories
    are distinguished); otherwise falls back to basename comparison.
    """
    item_path = item.get("source_path")
    if item_path and ("/" in span.file or "\\" in span.file):
        return Path(item_path).as_posix() == Path(span.file).as_posix()
    item_name = item_path or item.get("file_name") or ""
    return Path(item_name).name == Path(span.file).name


def _item_range(item: dict) -> tuple[int, int] | None:
    start, end = item.get("start_line"), item.get("end_line")
    if start is None or end is None:
        return None
    return int(start), int(end)


def _has_range(span: Span) -> bool:
    return span.start_line is not None and span.end_line is not None


def _line_tokens(weights: LineWeights, file: str, start: int, end: int) -> int:
    """Whitespace tokens on lines ``start..end`` (inclusive, 1-indexed) of ``file``."""
    per_line = weights[Path(file).name]
    return sum(per_line[start - 1 : end])


def _spans(query: Query, grade: int) -> list[Span]:
    return query.decisive_spans if grade == DECISIVE else query.supportive_spans


def _graded_spans(
    results: list[QueryResult],
    grade: int,
    span_filter: Callable[[Span], bool] | None,
) -> list[tuple[QueryResult, Span]]:
    """Ranged spans of ``grade`` (optionally filtered), paired with their result."""
    return [
        (r, span)
        for r in results
        for span in _spans(r.query, grade)
        if _has_range(span) and (span_filter is None or span_filter(span))
    ]


def coverage_at_k(
    results: list[QueryResult],
    k: int,
    weights: LineWeights,
    grade: int = DECISIVE,
    span_filter: Callable[[Span], bool] | None = None,
) -> float:
    """Did the answer arrive? Fraction of gold-span tokens present in the top-k.

    For each span: the tokens of its lines covered by the union of same-file
    top-k items, summed over all spans and divided by the total tokens across
    all spans (so longer answers weigh more). An answer that arrives split
    across two chunks still counts in full — the answer did arrive — which
    keeps the metric fair to chunkers that fragment. This replaces
    binary-overlap recall, which counted a span as fully retrieved when a
    chunk overlapped even one of its lines.
    """
    covered_total = 0
    span_total = 0
    for r, span in _graded_spans(results, grade, span_filter):
        assert span.start_line is not None and span.end_line is not None
        covered: set[int] = set()
        for item in r.retrieved[:k]:
            rng = _item_range(item)
            if rng is None or not _same_file(item, span):
                continue
            lo = max(rng[0], span.start_line)
            hi = min(rng[1], span.end_line)
            covered.update(range(lo, hi + 1))
        per_line = weights[Path(span.file).name]
        span_total += sum(per_line[span.start_line - 1 : span.end_line])
        covered_total += sum(per_line[line - 1] for line in covered)
    return covered_total / span_total if span_total else 0.0


def concentration_at_k(
    results: list[QueryResult],
    k: int,
    weights: LineWeights,
) -> float:
    """What fraction of all retrieved tokens was answer? Mean over queries.

    Per query: tokens of retrieved lines that fall inside any of that query's
    gold spans (decisive or supportive — supportive context is labelled
    useful, so it is not noise) divided by all tokens retrieved, averaged over
    queries. Lines retrieved twice by overlapping items count twice, because
    the agent reads them twice. This is the metric that penalises retrieving
    far more text than the answer needs: a whole-file chunk scores
    near-perfect coverage and near-zero concentration.
    """
    fractions: list[float] = []
    for r in results:
        spans = [s for s in (r.query.decisive_spans + r.query.supportive_spans) if _has_range(s)]
        useful = 0
        total = 0
        for item in r.retrieved[:k]:
            rng = _item_range(item)
            if rng is None:
                continue
            file_name = item.get("source_path") or item.get("file_name") or ""
            key = Path(file_name).name
            if key not in weights:
                continue
            total += _line_tokens(weights, key, rng[0], rng[1])
            useful_lines: set[int] = set()
            for span in spans:
                assert span.start_line is not None and span.end_line is not None
                if not _same_file(item, span):
                    continue
                lo = max(rng[0], span.start_line)
                hi = min(rng[1], span.end_line)
                useful_lines.update(range(lo, hi + 1))
            useful += sum(weights[key][line - 1] for line in useful_lines)
        if total:
            fractions.append(useful / total)
    return sum(fractions) / len(fractions) if fractions else 0.0


def wholeness_at_k(
    results: list[QueryResult],
    k: int,
    grade: int = DECISIVE,
    span_filter: Callable[[Span], bool] | None = None,
) -> float:
    """Did each answer arrive in one piece? Contained / overlapped.

    Among gold spans that any top-k item overlaps, the fraction fully
    contained within a *single* item. This catches a failure the other two
    metrics miss in combination: a chunk covering half an answer looks fine to
    both, but an agent reading half a function has no signal that the other
    half is missing. Conditional on overlap, so retrieving nothing scores 0
    (empty denominator), not a spurious 1. Uses line ranges only — containment
    needs no token weights. Read alongside coverage: wholeness alone says
    nothing about what was missed entirely.
    """
    overlapped = 0
    contained = 0
    for r, span in _graded_spans(results, grade, span_filter):
        assert span.start_line is not None and span.end_line is not None
        hit = False
        whole = False
        for item in r.retrieved[:k]:
            rng = _item_range(item)
            if rng is None or not _same_file(item, span):
                continue
            if rng[0] <= span.end_line and span.start_line <= rng[1]:
                hit = True
                if rng[0] <= span.start_line and span.end_line <= rng[1]:
                    whole = True
        if hit:
            overlapped += 1
            contained += int(whole)
    return contained / overlapped if overlapped else 0.0


def count_overlapped_spans(
    results: list[QueryResult],
    k: int,
    grade: int = DECISIVE,
    span_filter: Callable[[Span], bool] | None = None,
) -> int:
    """Count gold spans overlapped by any top-k item — wholeness's denominator."""
    overlapped = 0
    for r, span in _graded_spans(results, grade, span_filter):
        assert span.start_line is not None and span.end_line is not None
        for item in r.retrieved[:k]:
            rng = _item_range(item)
            if rng is None or not _same_file(item, span):
                continue
            if rng[0] <= span.end_line and span.start_line <= rng[1]:
                overlapped += 1
                break
    return overlapped


def count_spans(
    results: list[QueryResult],
    grade: int = DECISIVE,
    span_filter: Callable[[Span], bool] | None = None,
) -> int:
    """Count ranged gold spans of ``grade`` (optionally filtered) — coverage's denominator."""
    return len(_graded_spans(results, grade, span_filter))
