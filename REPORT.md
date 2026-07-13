# code-rag retrieval evaluation
## Setup

**Corpus** (corpus/): 16 files indexed.

| ext | files | lines |
|---|---|---|
| `.lean` | 4 | 1998 |
| `.md` | 4 | 4759 |
| `.py` | 8 | 2497 |

**Golden dataset**: 100 hand-written queries — by tool: lookup_index: 6, search_code: 46, search_documents: 48. Decisive gold spans by content type: `.lean`: 40, `.md`: 46, `.py`: 47.

**Configurations**: `ast+enrich[llama-4-scout]` · `ast+enrich[gemma-4-31b-it]` · `ast+enrich[deepseek-v4-flash]` · `ast` · `line-based` · `whole-file (canary)`. **Embedding model**: `qwen/qwen3-embedding-8b`. **Enrichment model(s)**: `meta-llama/llama-4-scout` · `google/gemma-4-31b-it` · `deepseek/deepseek-v4-flash`. **Retrieval**: top-k evaluated at 3, 5, 7, 10.

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

## Results
### Coverage — did the answer arrive?

| coverage | ast+enrich[llama-4-scout] | ast+enrich[gemma-4-31b-it] | ast+enrich[deepseek-v4-flash] | ast | line-based | whole-file (canary) |
|---|---|---|---|---|---|---|
| coverage@3 (decisive) | 0.678 | 0.696 | 0.689 | 0.643 | 0.678 | 0.851 |
| coverage@3 (supportive) | 0.685 | 0.675 | 0.698 | 0.640 | 0.612 | 0.893 |
| coverage@5 (decisive) | 0.780 | 0.766 | 0.791 | 0.734 | 0.770 | 0.906 |
| coverage@5 (supportive) | 0.751 | 0.748 | 0.767 | 0.726 | 0.752 | 0.926 |
| coverage@7 (decisive) | 0.795 | 0.805 | 0.836 | 0.778 | 0.820 | 0.938 |
| coverage@7 (supportive) | 0.786 | 0.821 | 0.847 | 0.812 | 0.800 | 0.926 |
| coverage@10 (decisive) | 0.835 | 0.855 | 0.840 | 0.814 | 0.831 | 0.971 |
| coverage@10 (supportive) | 0.834 | 0.828 | 0.854 | 0.840 | 0.829 | 0.954 |

### Coverage by content type (decisive spans)

| content-type | ast+enrich[llama-4-scout] | ast+enrich[gemma-4-31b-it] | ast+enrich[deepseek-v4-flash] | ast | line-based | whole-file (canary) |
|---|---|---|---|---|---|---|
| .lean coverage@3 (n=40) | 0.499 | 0.501 | 0.519 | 0.457 | 0.622 | 0.744 |
| .lean coverage@5 (n=40) | 0.563 | 0.509 | 0.604 | 0.542 | 0.668 | 0.788 |
| .lean coverage@7 (n=40) | 0.597 | 0.615 | 0.604 | 0.616 | 0.720 | 0.816 |
| .lean coverage@10 (n=40) | 0.597 | 0.636 | 0.621 | 0.660 | 0.720 | 0.833 |
| .md coverage@3 (n=46) | 0.723 | 0.730 | 0.733 | 0.699 | 0.737 | 0.883 |
| .md coverage@5 (n=46) | 0.846 | 0.818 | 0.837 | 0.805 | 0.841 | 0.924 |
| .md coverage@7 (n=46) | 0.846 | 0.843 | 0.905 | 0.838 | 0.893 | 0.965 |
| .md coverage@10 (n=46) | 0.905 | 0.908 | 0.905 | 0.869 | 0.900 | 1.000 |
| .py coverage@3 (n=47) | 0.686 | 0.763 | 0.691 | 0.617 | 0.509 | 0.836 |
| .py coverage@5 (n=47) | 0.750 | 0.828 | 0.806 | 0.654 | 0.599 | 0.958 |
| .py coverage@7 (n=47) | 0.804 | 0.851 | 0.806 | 0.712 | 0.637 | 0.963 |
| .py coverage@10 (n=47) | 0.811 | 0.874 | 0.811 | 0.761 | 0.678 | 1.000 |

### Concentration — how much of what the agent read was answer?

| concentration | ast+enrich[llama-4-scout] | ast+enrich[gemma-4-31b-it] | ast+enrich[deepseek-v4-flash] | ast | line-based | whole-file (canary) |
|---|---|---|---|---|---|---|
| concentration@3 | 0.243 | 0.249 | 0.258 | 0.223 | 0.141 | 0.016 |
| concentration@5 | 0.171 | 0.179 | 0.187 | 0.155 | 0.100 | 0.008 |
| concentration@7 | 0.129 | 0.137 | 0.145 | 0.119 | 0.080 | 0.005 |
| concentration@10 | 0.094 | 0.099 | 0.098 | 0.089 | 0.056 | 0.003 |

### Wholeness — did covered answers arrive in one piece?

| wholeness | ast+enrich[llama-4-scout] | ast+enrich[gemma-4-31b-it] | ast+enrich[deepseek-v4-flash] | ast | line-based | whole-file (canary) |
|---|---|---|---|---|---|---|
| overall wholeness@3 | 0.961 (n=77) | 0.952 (n=84) | 0.951 (n=81) | 0.946 (n=74) | 0.899 (n=79) | 1.000 (n=102) |
| overall wholeness@5 | 0.944 (n=90) | 0.934 (n=91) | 0.937 (n=95) | 0.952 (n=83) | 0.899 (n=89) | 1.000 (n=113) |
| overall wholeness@7 | 0.947 (n=95) | 0.938 (n=97) | 0.938 (n=96) | 0.945 (n=91) | 0.887 (n=97) | 1.000 (n=118) |
| overall wholeness@10 | 0.948 (n=97) | 0.941 (n=101) | 0.939 (n=98) | 0.938 (n=96) | 0.888 (n=98) | 1.000 (n=125) |
| .lean wholeness@3 | 0.882 (n=17) | 0.824 (n=17) | 0.882 (n=17) | 0.812 (n=16) | 0.917 (n=24) | 1.000 (n=28) |
| .lean wholeness@5 | 0.895 (n=19) | 0.833 (n=18) | 0.857 (n=21) | 0.850 (n=20) | 0.920 (n=25) | 1.000 (n=30) |
| .lean wholeness@7 | 0.905 (n=21) | 0.857 (n=21) | 0.857 (n=21) | 0.864 (n=22) | 0.923 (n=26) | 1.000 (n=31) |
| .lean wholeness@10 | 0.905 (n=21) | 0.864 (n=22) | 0.864 (n=22) | 0.870 (n=23) | 0.923 (n=26) | 1.000 (n=32) |
| .md wholeness@3 | 0.970 (n=33) | 0.971 (n=34) | 0.943 (n=35) | 0.971 (n=34) | 0.941 (n=34) | 1.000 (n=37) |
| .md wholeness@5 | 0.925 (n=40) | 0.921 (n=38) | 0.925 (n=40) | 0.973 (n=37) | 0.949 (n=39) | 1.000 (n=40) |
| .md wholeness@7 | 0.925 (n=40) | 0.925 (n=40) | 0.927 (n=41) | 0.950 (n=40) | 0.951 (n=41) | 1.000 (n=43) |
| .md wholeness@10 | 0.927 (n=41) | 0.929 (n=42) | 0.927 (n=41) | 0.929 (n=42) | 0.951 (n=41) | 1.000 (n=46) |
| .py wholeness@3 | 1.000 (n=27) | 1.000 (n=33) | 1.000 (n=29) | 1.000 (n=24) | 0.810 (n=21) | 1.000 (n=37) |
| .py wholeness@5 | 1.000 (n=31) | 1.000 (n=35) | 1.000 (n=34) | 1.000 (n=26) | 0.800 (n=25) | 1.000 (n=43) |
| .py wholeness@7 | 1.000 (n=34) | 1.000 (n=36) | 1.000 (n=34) | 1.000 (n=29) | 0.767 (n=30) | 1.000 (n=44) |
| .py wholeness@10 | 1.000 (n=35) | 1.000 (n=37) | 1.000 (n=35) | 1.000 (n=31) | 0.774 (n=31) | 1.000 (n=47) |

### Chunk geometry

| geometry | ast+enrich[llama-4-scout] | ast+enrich[gemma-4-31b-it] | ast+enrich[deepseek-v4-flash] | ast | line-based | whole-file (canary) |
|---|---|---|---|---|---|---|
| chunks | 432 | 432 | 432 | 432 | 238 | 15 |
| mean lines | 22.6 | 22.6 | 22.6 | 22.6 | 38.9 | 616.9 |
| median lines | 12 | 12 | 12 | 12 | 40 | 467 |
| max lines | 239 | 239 | 239 | 239 | 40 | 1672 |

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