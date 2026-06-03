"""Render evaluation metrics into Markdown tables and a full academic report.

Tables are GitHub-flavoured Markdown so the generated report drops straight into
a README. Metric *definitions* are static prose (``_METRICS_SECTION``); the
numbers are computed from the run, so the report never goes stale.
"""

from collections.abc import Callable
from pathlib import Path
from statistics import mean, median

from ..models import Chunk
from .dataset import Span
from .metrics import (
    QueryResult,
    count_ranged_spans,
    count_spans,
    hit_rate,
    mean_iou_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def _md_table(corner: str, col_labels: list[str], rows: list[tuple[str, list[str]]]) -> str:
    """Render a Markdown table: first column ``corner``, then one per configuration."""
    header = "| " + " | ".join([corner, *col_labels]) + " |"
    sep = "|" + "---|" * (len(col_labels) + 1)
    body = ["| " + " | ".join([name, *cells]) + " |" for name, cells in rows]
    return "\n".join([header, sep, *body])


def _ext_predicate(ext: str) -> Callable[[Span], bool]:
    def in_ext(span: Span, _ext: str = ext) -> bool:
        return Path(span.file).suffix == _ext

    return in_ext


def _content_types(results: list[QueryResult]) -> list[str]:
    """Distinct file extensions appearing in any decisive gold span, sorted."""
    exts = {
        Path(s.file).suffix for r in results for s in r.query.decisive_spans if Path(s.file).suffix
    }
    return sorted(exts)


def build_report(
    results_by_label: dict[str, list[QueryResult]],
    top_k_values: list[int],
) -> str:
    """Overall metric × configuration table (hit-rate, recall, NDCG, MRR)."""
    labels = list(results_by_label)
    rows: list[tuple[str, list[str]]] = []

    def row(name: str, fn: Callable[..., float]) -> None:
        rows.append((name, [f"{fn(results_by_label[c]):.3f}" for c in labels]))

    row("hit_rate@decisive", lambda r: hit_rate(r, 2))
    for k in top_k_values:
        row(f"recall@{k} (decisive)", lambda r, k=k: recall_at_k(r, k, 2))
        row(f"recall@{k} (supportive)", lambda r, k=k: recall_at_k(r, k, 1))
        row(f"ndcg@{k}", lambda r, k=k: ndcg_at_k(r, k))
        row(f"mrr@{k}", lambda r, k=k: mrr_at_k(r, k))
    return _md_table("metric", labels, rows)


def build_breakdown(
    results_by_label: dict[str, list[QueryResult]],
    top_k_values: list[int],
) -> str:
    """Decisive recall@k sliced by gold-span content type, per configuration."""
    labels = list(results_by_label)
    any_results = next(iter(results_by_label.values()))
    rows: list[tuple[str, list[str]]] = []
    for ext in _content_types(any_results):
        pred = _ext_predicate(ext)
        n = count_spans(any_results, 2, pred)
        for k in top_k_values:
            rows.append(
                (
                    f"{ext} recall@{k} (n={n})",
                    [f"{recall_at_k(results_by_label[c], k, 2, pred):.3f}" for c in labels],
                )
            )
    return _md_table("content-type", labels, rows)


def build_iou_table(
    results_by_label: dict[str, list[QueryResult]],
    top_k_values: list[int],
) -> str:
    """Mean line-range IoU@k (decisive), overall and per content type, with sample n."""
    labels = list(results_by_label)
    any_results = next(iter(results_by_label.values()))
    rows: list[tuple[str, list[str]]] = []

    n_all = count_ranged_spans(any_results, 2)
    for k in top_k_values:
        rows.append(
            (
                f"overall IoU@{k} (n={n_all})",
                [f"{mean_iou_at_k(results_by_label[c], k, 2):.3f}" for c in labels],
            )
        )
    for ext in _content_types(any_results):
        pred = _ext_predicate(ext)
        n = count_ranged_spans(any_results, 2, pred)
        for k in top_k_values:
            rows.append(
                (
                    f"{ext} IoU@{k} (n={n})",
                    [f"{mean_iou_at_k(results_by_label[c], k, 2, pred):.3f}" for c in labels],
                )
            )
    return _md_table("localization", labels, rows)


def chunk_geometry(chunks: list[Chunk]) -> dict[str, float]:
    """Count and line-size distribution of a configuration's chunks."""
    sizes = [c.end_line - c.start_line + 1 for c in chunks]
    if not sizes:
        return {"count": 0, "mean_lines": 0.0, "median_lines": 0.0, "max_lines": 0}
    return {
        "count": len(sizes),
        "mean_lines": mean(sizes),
        "median_lines": median(sizes),
        "max_lines": max(sizes),
    }


def build_geometry_table(geometry_by_label: dict[str, dict[str, float]]) -> str:
    """Chunk-geometry (count / mean / median / max lines) × configuration table."""
    labels = list(geometry_by_label)
    spec = [
        ("chunks", "count", "{:.0f}"),
        ("mean lines", "mean_lines", "{:.1f}"),
        ("median lines", "median_lines", "{:.0f}"),
        ("max lines", "max_lines", "{:.0f}"),
    ]
    rows = [
        (name, [fmt.format(geometry_by_label[c][key]) for c in labels]) for name, key, fmt in spec
    ]
    return _md_table("geometry", labels, rows)


_METRICS_SECTION = """\
## Metrics

Relevance is hand-labelled at the **span** level (a file region identified by
symbol and/or line range), graded *decisive* (2 — directly answers the query) or
*supportive* (1 — helpful context). A retrieved item **hits** a span when it is
in the same file and either overlaps the span's line range or matches its symbol.
All scores are in [0, 1].

- **hit-rate@decisive** — fraction of queries with at least one decisive-span hit
  anywhere in the retrieved set. A coarse "did we surface anything relevant" signal.
- **recall@k (decisive / supportive)** — fraction of all gold spans of that grade
  found within the top-k items, micro-averaged over spans. *Primary metric*: the
  agent consumes the top-k as a set, so coverage matters more than rank.
- **NDCG@k** — normalised discounted cumulative gain over graded span *coverage*
  (decisive=2, supportive=1), each span credited once at the rank it is first
  covered. Rewards ranking relevant spans near the top.
- **MRR@k** — mean reciprocal rank of the first decisive hit within the top-k.
- **line-range IoU@k** — mean over decisive spans (that carry a line range) of the
  best intersection-over-union between the span and any top-k item. Measures
  *localization precision*: a chunk tightly bounding the span scores near 1, a
  large window that merely overlaps it scores near 0. Complements recall, which is
  satisfied by *any* overlap. Symbol-only spans are excluded (n reported per row).
- **chunk geometry** — the count and line-size distribution of chunks each
  configuration produces. Context, not a score: overlap-based recall rewards
  larger chunks, so geometry and IoU together reveal whether a configuration wins
  by precision or merely by coarseness.
"""

_CAVEATS_SECTION = """\
## Caveats

- **Overlap-based recall favours coarse chunks.** At equal k, larger retrieval
  units overlap more gold-span lines without being more useful (in the limit, one
  whole-file chunk scores recall ≈ 1). Read recall alongside IoU and chunk
  geometry, never alone.
- **Small per-type sample sizes.** With tens of spans per content type, treat
  differences below ~0.05 as noise. Confidence intervals are not yet reported.
- **`lookup_index` is substring matching** over a static concept index, not
  semantic search; natural-language queries pinned to it may under-retrieve
  regardless of the embedding model.
"""


def build_markdown_report(
    results_by_label: dict[str, list[QueryResult]],
    geometry_by_label: dict[str, dict[str, float]],
    top_k_values: list[int],
    setup_md: str,
) -> str:
    """Assemble the full academic-style Markdown report (setup, metrics, results)."""
    parts = [
        "# code-rag retrieval evaluation",
        setup_md,
        _METRICS_SECTION,
        "## Results",
        "### Overall (micro-averaged over all queries)\n",
        build_report(results_by_label, top_k_values),
        "\n### Recall@k by content type (decisive spans)\n",
        build_breakdown(results_by_label, top_k_values),
        "\n### Chunk geometry\n",
        build_geometry_table(geometry_by_label),
        "\n### Localization — line-range IoU@k (decisive spans)\n",
        build_iou_table(results_by_label, top_k_values),
        "",
        _CAVEATS_SECTION,
    ]
    return "\n".join(parts)
