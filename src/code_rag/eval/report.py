"""Render evaluation results into Markdown tables and a complete report.

Tables use GitHub-flavoured Markdown so the report is readable on GitHub
without conversion. The metric definitions are fixed prose
(``_METRICS_SECTION``); the numbers are computed from the run, so the
definitions and the results in one report always describe the same code.
"""

from collections.abc import Callable
from pathlib import Path
from statistics import mean, median

from ..models import Chunk
from .dataset import Span
from .metrics import (
    LineWeights,
    QueryResult,
    concentration_at_k,
    count_overlapped_spans,
    count_spans,
    coverage_at_k,
    wholeness_at_k,
)

CANARY_LABEL = "whole-file (canary)"


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


def build_coverage_table(
    results_by_label: dict[str, list[QueryResult]],
    weights: LineWeights,
    top_k_values: list[int],
) -> str:
    """Overall coverage@k (decisive + supportive) × configuration."""
    labels = list(results_by_label)
    rows: list[tuple[str, list[str]]] = []
    for k in top_k_values:
        for grade, name in ((2, "decisive"), (1, "supportive")):
            rows.append(
                (
                    f"coverage@{k} ({name})",
                    [
                        f"{coverage_at_k(results_by_label[c], k, weights, grade):.3f}"
                        for c in labels
                    ],
                )
            )
    return _md_table("coverage", labels, rows)


def build_coverage_breakdown(
    results_by_label: dict[str, list[QueryResult]],
    weights: LineWeights,
    top_k_values: list[int],
) -> str:
    """Decisive coverage@k sliced by gold-span content type, per configuration."""
    labels = list(results_by_label)
    any_results = next(iter(results_by_label.values()))
    rows: list[tuple[str, list[str]]] = []
    for ext in _content_types(any_results):
        pred = _ext_predicate(ext)
        n = count_spans(any_results, 2, pred)
        for k in top_k_values:
            rows.append(
                (
                    f"{ext} coverage@{k} (n={n})",
                    [
                        f"{coverage_at_k(results_by_label[c], k, weights, 2, pred):.3f}"
                        for c in labels
                    ],
                )
            )
    return _md_table("content-type", labels, rows)


def build_concentration_table(
    results_by_label: dict[str, list[QueryResult]],
    weights: LineWeights,
    top_k_values: list[int],
) -> str:
    """Concentration@k × configuration."""
    labels = list(results_by_label)
    rows = [
        (
            f"concentration@{k}",
            [f"{concentration_at_k(results_by_label[c], k, weights):.3f}" for c in labels],
        )
        for k in top_k_values
    ]
    return _md_table("concentration", labels, rows)


def build_wholeness_table(
    results_by_label: dict[str, list[QueryResult]],
    top_k_values: list[int],
) -> str:
    """Wholeness@k (decisive), overall and per content type.

    Wholeness is conditional on overlap, so its denominator varies per
    configuration and per k; each cell carries its own n.
    """
    labels = list(results_by_label)
    any_results = next(iter(results_by_label.values()))

    def cell(c: str, k: int, pred: Callable[[Span], bool] | None) -> str:
        results = results_by_label[c]
        n = count_overlapped_spans(results, k, 2, pred)
        return f"{wholeness_at_k(results, k, 2, pred):.3f} (n={n})"

    rows: list[tuple[str, list[str]]] = []
    for k in top_k_values:
        rows.append((f"overall wholeness@{k}", [cell(c, k, None) for c in labels]))
    for ext in _content_types(any_results):
        pred = _ext_predicate(ext)
        for k in top_k_values:
            rows.append((f"{ext} wholeness@{k}", [cell(c, k, pred) for c in labels]))
    return _md_table("wholeness", labels, rows)


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


def canary_check(
    results_by_label: dict[str, list[QueryResult]],
    weights: LineWeights,
    top_k_values: list[int],
    canary_label: str = CANARY_LABEL,
) -> list[str]:
    """Check that the whole-file baseline never scores best overall.

    The whole-file configuration indexes each file as one chunk. Retrieving
    whole files maximises coverage while returning mostly irrelevant text, so
    a working metric suite must never rank it best on coverage *and*
    concentration at the same k. If that happens, the metrics have stopped
    penalising oversized chunks; return one warning per k where it occurred.
    """
    if canary_label not in results_by_label:
        return []
    warnings: list[str] = []
    others = [label for label in results_by_label if label != canary_label]
    for k in top_k_values:
        canary_cov = coverage_at_k(results_by_label[canary_label], k, weights)
        canary_con = concentration_at_k(results_by_label[canary_label], k, weights)
        if all(
            canary_cov > coverage_at_k(results_by_label[c], k, weights)
            and canary_con > concentration_at_k(results_by_label[c], k, weights)
            for c in others
        ):
            warnings.append(
                f"**WARNING (k={k})**: the whole-file baseline scored best on both "
                f"coverage and concentration. These metrics are designed to penalise "
                f"oversized chunks and did not. Do not act on this report until the "
                f"metric implementation is fixed."
            )
    return warnings


_METRICS_SECTION = """\
## Metrics

**How relevance is labelled.** Every query is paired with hand-labelled spans:
regions of a corpus file identified by a line range. (A symbol name may be
recorded alongside a span, but it is documentation only — scoring never reads
it.) A span carries one of two grades: *decisive* (it directly answers the
query) or *supportive* (it is useful context around an answer). Decisive spans
are labelled at the smallest region that states the answer. They are never
sized to match a chunker's output — a whole function or a whole section — so
no chunking strategy gets gold labels that equal its own chunks by
construction.

**How a retrieved chunk matches a span.** A retrieved chunk matches a span
when both point at the same file and their line ranges overlap. Symbol names
are ignored during scoring: only the AST chunkers record them, so scoring
through symbol names would favour those chunkers over strategies that have no
symbols to report.

**Why scores count tokens, not lines.** Every score is weighted by
whitespace-separated tokens per line, not by line count. In this corpus a
markdown "line" is often an entire paragraph while a Python line is around 40
characters, so line counts are not comparable across file types. All scores
fall in [0, 1].

**The three metrics.** The intended consumer of retrieval is an LLM agent. It
reads the whole top-k result set at once and pays — in context window and API
cost — for every token it reads. Retrieval can fail that consumer in three
distinct ways, and each metric measures one of them:

- **coverage@k** — did the answer arrive? For each gold span: the fraction of
  its tokens that appear anywhere in the top-k results, summed over all spans
  and divided by the total tokens across all spans (so longer answers weigh
  more). An answer that arrives split across two chunks still counts in full.
  This replaces binary-overlap recall, which counted a span as fully retrieved
  when a chunk overlapped even one of its lines.
- **concentration@k** — how much of what was read was answer? Per query:
  tokens of retrieved lines that fall inside any of that query's spans
  (decisive or supportive) divided by all tokens retrieved, averaged over
  queries. Lines retrieved twice count twice, because the agent reads them
  twice. This is the metric that penalises retrieving far more text than the
  answer needs: a whole-file chunk scores near-perfect coverage and near-zero
  concentration.
- **wholeness@k** — did each answer arrive in one piece? Among the decisive
  spans that any top-k result overlaps, the fraction fully contained inside a
  single result. The sample size varies per cell and is reported next to each
  score. This catches a failure the other two metrics miss in combination:
  two chunks that each cover half an answer score well on both coverage and
  concentration, but an agent reading half a function has no signal that the
  other half is missing.
- **chunk geometry** — chunk count and line-size distribution per
  configuration. Context for reading the other tables, not a score.

Ranking metrics (NDCG, MRR) are not reported. They measure whether the answer
arrived *early* in the result list, and an agent that reads the whole top-k as
one set does not care about position. Sensitivity to result-set size is
reported instead, by computing every metric at each k.

**The whole-file baseline (the "canary").** One configuration in every run
indexes each file as a single chunk. It exists to test the metrics, not to
compete: retrieving whole files trivially maximises coverage while burying
answers in unrelated text, so a working metric suite must never rank it best
overall. If it ever scores highest on both coverage and concentration at the
same k, the metrics have stopped penalising oversized chunks, and the report
prints a warning saying it should not be trusted.
"""

_CAVEATS_SECTION = """\
## Caveats

- **Wholeness depends on what was found.** Its denominator is only the spans a
  configuration overlapped at all, so a configuration that misses more answers
  is graded on fewer of them. Read wholeness together with coverage, never on
  its own.
- **Per-type sample sizes are small.** With roughly forty to fifty decisive
  spans per file type, treat differences smaller than about 0.05 as noise.
  Confidence intervals are not yet reported.
- **`lookup_index` is exact substring matching** against a prebuilt concept
  index, not semantic search. Natural-language queries assigned to that tool
  can return few or no results regardless of the embedding model.
"""


def build_markdown_report(
    results_by_label: dict[str, list[QueryResult]],
    geometry_by_label: dict[str, dict[str, float]],
    top_k_values: list[int],
    setup_md: str,
    weights: LineWeights,
) -> str:
    """Assemble the complete Markdown report: setup, metric definitions, results, caveats."""
    parts = [
        "# code-rag retrieval evaluation",
        setup_md,
        _METRICS_SECTION,
        "## Results",
        "### Coverage — did the answer arrive?\n",
        build_coverage_table(results_by_label, weights, top_k_values),
        "\n### Coverage by content type (decisive spans)\n",
        build_coverage_breakdown(results_by_label, weights, top_k_values),
        "\n### Concentration — how much of what the agent read was answer?\n",
        build_concentration_table(results_by_label, weights, top_k_values),
        "\n### Wholeness — did covered answers arrive in one piece?\n",
        build_wholeness_table(results_by_label, top_k_values),
        "\n### Chunk geometry\n",
        build_geometry_table(geometry_by_label),
    ]
    warnings = canary_check(results_by_label, weights, top_k_values)
    if warnings:
        parts += ["\n## Warning: metric self-check failed\n", "\n".join(warnings)]
    parts += ["", _CAVEATS_SECTION]
    return "\n".join(parts)
