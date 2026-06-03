"""
Retrieval metrics over span-level graded relevance.

A retrieved item (a chunk payload or pointer entry, normalised to a dict with
``file_name``/``start_line``/``end_line``/``symbol_name``) *hits* a gold span when
it is in the same file and either overlaps the span's line range or references
its symbol name. Primary metric is recall@k of decisive spans — Claude reads the
top-k as a set, so coverage matters more than rank.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .dataset import Query, Span

DECISIVE = 2
SUPPORTIVE = 1


@dataclass
class QueryResult:
    """A query paired with the ranked items retrieved for it."""

    query: Query
    retrieved: list[dict]  # rank-ordered; each has file_name/start_line/end_line/symbol_name


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


def _line_overlap(item: dict, span: Span) -> bool:
    if span.start_line is None or span.end_line is None:
        return False
    istart = item.get("start_line")
    iend = item.get("end_line")
    if istart is None or iend is None:
        return False
    return bool(istart <= span.end_line and span.start_line <= iend)


def line_iou(item: dict, span: Span) -> float:
    """Intersection-over-union of the item's and span's inclusive line ranges.

    Pure geometry (ignores symbol match): 0.0 for a different file, a missing
    range on either side, or no overlap. Rewards tightly-bounded retrieval — a
    40-line window over a 5-line span scores 5/40, a 6-line method over it 5/6.
    """
    if not _same_file(item, span):
        return 0.0
    if span.start_line is None or span.end_line is None:
        return 0.0
    istart, iend = item.get("start_line"), item.get("end_line")
    if istart is None or iend is None:
        return 0.0
    istart, iend = int(istart), int(iend)
    inter = min(iend, span.end_line) - max(istart, span.start_line) + 1
    if inter <= 0:
        return 0.0
    union = (iend - istart + 1) + (span.end_line - span.start_line + 1) - inter
    return inter / union


def _symbol_match(item: dict, span: Span) -> bool:
    if not span.symbol:
        return False
    name = item.get("symbol_name") or ""
    # Match "Widget" against "Widget" and "Widget.area" (and vice versa).
    return span.symbol == name or span.symbol in name or name in span.symbol


def item_hits_span(item: dict, span: Span) -> bool:
    """True when a retrieved item overlaps a gold span (line range or symbol)."""
    if not _same_file(item, span):
        return False
    return _line_overlap(item, span) or _symbol_match(item, span)


def _spans(query: Query, grade: int) -> list[Span]:
    return query.decisive_spans if grade == DECISIVE else query.supportive_spans


def hit_rate(results: list[QueryResult], grade: int = DECISIVE) -> float:
    """Fraction of queries with at least one top-result hit on a span of ``grade``."""
    considered = [r for r in results if _spans(r.query, grade)]
    if not considered:
        return 0.0
    hits = sum(
        any(any(item_hits_span(it, s) for s in _spans(r.query, grade)) for it in r.retrieved)
        for r in considered
    )
    return hits / len(considered)


def recall_at_k(
    results: list[QueryResult],
    k: int,
    grade: int = DECISIVE,
    span_filter: Callable[[Span], bool] | None = None,
) -> float:
    """Fraction of gold spans of ``grade`` found within the top-k items (micro).

    Pass ``span_filter`` to restrict the denominator to a subset of spans (e.g.
    only spans in ``.lean`` files), enabling per-content-type recall.
    """
    total = 0
    found = 0
    for r in results:
        topk = r.retrieved[:k]
        for span in _spans(r.query, grade):
            if span_filter is not None and not span_filter(span):
                continue
            total += 1
            if any(item_hits_span(it, span) for it in topk):
                found += 1
    return found / total if total else 0.0


def count_spans(
    results: list[QueryResult],
    grade: int = DECISIVE,
    span_filter: Callable[[Span], bool] | None = None,
) -> int:
    """Count gold spans of ``grade`` (optionally filtered) — the recall denominator."""
    return sum(
        1
        for r in results
        for span in _spans(r.query, grade)
        if span_filter is None or span_filter(span)
    )


def _has_range(span: Span) -> bool:
    return span.start_line is not None and span.end_line is not None


def mean_iou_at_k(
    results: list[QueryResult],
    k: int,
    grade: int = DECISIVE,
    span_filter: Callable[[Span], bool] | None = None,
) -> float:
    """Mean best line-range IoU over gold spans of ``grade`` within the top-k items.

    For each span (that has a line range), takes the largest ``line_iou`` among the
    top-k retrieved items, then averages. Measures *localization precision* — how
    tightly retrieval bounds the gold region — complementing recall's coverage.
    Spans without a line range are excluded (IoU is undefined for them), so this
    denominator can be smaller than recall's; report it separately.
    """
    scores: list[float] = []
    for r in results:
        topk = r.retrieved[:k]
        for span in _spans(r.query, grade):
            if not _has_range(span):
                continue
            if span_filter is not None and not span_filter(span):
                continue
            scores.append(max((line_iou(it, span) for it in topk), default=0.0))
    return sum(scores) / len(scores) if scores else 0.0


def count_ranged_spans(
    results: list[QueryResult],
    grade: int = DECISIVE,
    span_filter: Callable[[Span], bool] | None = None,
) -> int:
    """Count gold spans of ``grade`` that have a line range — the IoU denominator."""
    return sum(
        1
        for r in results
        for span in _spans(r.query, grade)
        if _has_range(span) and (span_filter is None or span_filter(span))
    )


def _dcg(relevances: list[int]) -> float:
    return float(sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances)))


def ndcg_at_k(results: list[QueryResult], k: int) -> float:
    """NDCG@k over graded span *coverage* (decisive=2, supportive=1), averaged over queries.

    Each gold span is credited at most once — at the rank of the first retrieved
    item that covers it. Without this, overlapping chunks that all hit one span
    (e.g. naive line windows) accumulate gains beyond the ideal and push NDCG
    above 1. Crediting per span keeps the achieved gains a subset of the ideal
    multiset, so NDCG stays in [0, 1] and rewards rank/coverage, not redundancy.
    """
    scores = []
    for r in results:
        decisive, supportive = r.query.decisive_spans, r.query.supportive_spans
        seen_dec: set[int] = set()
        seen_sup: set[int] = set()
        gains: list[int] = []
        for item in r.retrieved[:k]:
            grade = 0
            for i, span in enumerate(decisive):
                if i not in seen_dec and item_hits_span(item, span):
                    seen_dec.add(i)
                    grade = DECISIVE
                    break
            else:
                for i, span in enumerate(supportive):
                    if i not in seen_sup and item_hits_span(item, span):
                        seen_sup.add(i)
                        grade = SUPPORTIVE
                        break
            gains.append(grade)
        ideal = sorted([DECISIVE] * len(decisive) + [SUPPORTIVE] * len(supportive), reverse=True)
        idcg = _dcg(ideal[:k])
        if idcg == 0:
            continue
        scores.append(_dcg(gains) / idcg)
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(results: list[QueryResult], k: int) -> float:
    """Mean reciprocal rank of the first decisive-span hit within the top-k."""
    considered = [r for r in results if r.query.decisive_spans]
    if not considered:
        return 0.0
    total = 0.0
    for r in considered:
        for rank, item in enumerate(r.retrieved[:k], start=1):
            if any(item_hits_span(item, s) for s in r.query.decisive_spans):
                total += 1.0 / rank
                break
    return total / len(considered)
