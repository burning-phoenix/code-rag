# code-rag retrieval evaluation
## Setup

**Corpus** (corpus/): 16 files indexed.

| ext | files | lines |
|---|---|---|
| `.lean` | 4 | 1998 |
| `.md` | 4 | 4759 |
| `.py` | 8 | 2497 |

**Golden dataset**: 100 hand-written queries — by tool: lookup_index: 6, search_code: 46, search_documents: 48. Decisive gold spans by content type: `.lean`: 39, `.md`: 43, `.py`: 44.

**Configurations**: `ast+enrich` · `ast` · `line-based`. **Embedding model**: `qwen/qwen3-embedding-8b`. **Retrieval**: top-k swept over 3, 5, 7, 10.

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

## Results
### Overall (micro-averaged over all queries)

| metric | ast+enrich | ast | line-based |
|---|---|---|---|
| hit_rate@decisive | 0.890 | 0.850 | 0.860 |
| recall@3 (decisive) | 0.659 | 0.595 | 0.627 |
| recall@3 (supportive) | 0.487 | 0.487 | 0.436 |
| ndcg@3 | 0.658 | 0.601 | 0.651 |
| mrr@3 | 0.677 | 0.618 | 0.678 |
| recall@5 (decisive) | 0.706 | 0.627 | 0.683 |
| recall@5 (supportive) | 0.538 | 0.564 | 0.513 |
| ndcg@5 | 0.680 | 0.619 | 0.676 |
| mrr@5 | 0.685 | 0.624 | 0.689 |
| recall@7 (decisive) | 0.738 | 0.683 | 0.714 |
| recall@7 (supportive) | 0.564 | 0.615 | 0.564 |
| ndcg@7 | 0.694 | 0.642 | 0.686 |
| mrr@7 | 0.691 | 0.634 | 0.693 |
| recall@10 (decisive) | 0.762 | 0.722 | 0.722 |
| recall@10 (supportive) | 0.590 | 0.641 | 0.590 |
| ndcg@10 | 0.700 | 0.652 | 0.689 |
| mrr@10 | 0.692 | 0.636 | 0.694 |

### Recall@k by content type (decisive spans)

| content-type | ast+enrich | ast | line-based |
|---|---|---|---|
| .lean recall@3 (n=39) | 0.564 | 0.538 | 0.615 |
| .lean recall@5 (n=39) | 0.590 | 0.564 | 0.641 |
| .lean recall@7 (n=39) | 0.641 | 0.615 | 0.641 |
| .lean recall@10 (n=39) | 0.667 | 0.615 | 0.641 |
| .md recall@3 (n=43) | 0.814 | 0.767 | 0.744 |
| .md recall@5 (n=43) | 0.860 | 0.814 | 0.860 |
| .md recall@7 (n=43) | 0.884 | 0.860 | 0.884 |
| .md recall@10 (n=43) | 0.907 | 0.907 | 0.884 |
| .py recall@3 (n=44) | 0.591 | 0.477 | 0.523 |
| .py recall@5 (n=44) | 0.659 | 0.500 | 0.545 |
| .py recall@7 (n=44) | 0.682 | 0.568 | 0.614 |
| .py recall@10 (n=44) | 0.705 | 0.636 | 0.636 |

### Chunk geometry

| geometry | ast+enrich | ast | line-based |
|---|---|---|---|
| chunks | 432 | 432 | 238 |
| mean lines | 22.6 | 22.6 | 38.9 |
| median lines | 12 | 12 | 40 |
| max lines | 239 | 239 | 40 |

### Localization — line-range IoU@k (decisive spans)

| localization | ast+enrich | ast | line-based |
|---|---|---|---|
| overall IoU@3 (n=108) | 0.384 | 0.340 | 0.193 |
| overall IoU@5 (n=108) | 0.418 | 0.369 | 0.229 |
| overall IoU@7 (n=108) | 0.426 | 0.396 | 0.245 |
| overall IoU@10 (n=108) | 0.428 | 0.417 | 0.250 |
| .lean IoU@3 (n=21) | 0.439 | 0.356 | 0.224 |
| .lean IoU@5 (n=21) | 0.463 | 0.458 | 0.235 |
| .lean IoU@7 (n=21) | 0.490 | 0.515 | 0.239 |
| .lean IoU@10 (n=21) | 0.490 | 0.515 | 0.239 |
| .md IoU@3 (n=43) | 0.315 | 0.305 | 0.235 |
| .md IoU@5 (n=43) | 0.354 | 0.312 | 0.281 |
| .md IoU@7 (n=43) | 0.359 | 0.335 | 0.295 |
| .md IoU@10 (n=43) | 0.362 | 0.344 | 0.295 |
| .py IoU@3 (n=44) | 0.424 | 0.367 | 0.138 |
| .py IoU@5 (n=44) | 0.458 | 0.382 | 0.176 |
| .py IoU@7 (n=44) | 0.460 | 0.400 | 0.199 |
| .py IoU@10 (n=44) | 0.463 | 0.442 | 0.211 |

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
