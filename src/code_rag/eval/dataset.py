"""
Golden dataset: hand-written query → relevant-span pairs.

Relevance is labelled at the *span* level (file + line range, with an optional
symbol kept as documentation), not the chunk level, so the dataset stays valid
when chunking parameters change. Scoring is line-geometry only — symbols are
never consulted (see eval/metrics.py).

Labeling rules (every span must follow these):

1. **Decisive spans are answer-sized**: the minimal contiguous region that
   states the answer — never sized to a parser's unit (a whole function or
   section), so no chunker's native output equals gold by construction.
2. **Every decisive span carries a line range.** A rangeless span is
   unscorable under geometry-only matching.
3. **Every independently-sufficient location is decisive.** If two places each
   fully answer the query, both are labeled, so a retriever finding either is
   credited.
4. **Enclosing nodes and surrounding context are supportive** (grade 1) — the
   function around the answering lines, the section around the theorem.

The dataset is hand-written, never LLM-generated: LLM-authored queries would
share phrasing with LLM-authored enrichment summaries and bias that ablation
in its own favor.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

# Tools a query can target.
VALID_TOOLS = {"search_code", "search_documents", "lookup_index"}


@dataclass
class Span:
    """A relevant region of the corpus, identified by symbol and/or line range."""

    file: str
    symbol: str | None = None
    type: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        return cls(
            file=d["file"],
            symbol=d.get("symbol"),
            type=d.get("type"),
            start_line=d.get("start_line"),
            end_line=d.get("end_line"),
        )


@dataclass
class Query:
    """A natural-language query with graded relevant spans."""

    id: str
    query: str
    tool: str
    decisive_spans: list[Span] = field(default_factory=list)  # grade 2
    supportive_spans: list[Span] = field(default_factory=list)  # grade 1

    @classmethod
    def from_dict(cls, d: dict) -> "Query":
        tool = d.get("tool", "search_documents")
        if tool not in VALID_TOOLS:
            raise ValueError(f"Query {d.get('id')!r}: invalid tool {tool!r}")
        return cls(
            id=d["id"],
            query=d["query"],
            tool=tool,
            decisive_spans=[Span.from_dict(s) for s in d.get("decisive_spans", [])],
            supportive_spans=[Span.from_dict(s) for s in d.get("supportive_spans", [])],
        )


@dataclass
class GoldenDataset:
    """A corpus directory plus the hand-written queries evaluated against it."""

    corpus_dir: str
    queries: list[Query] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.queries)


def load_dataset(path: str | Path) -> GoldenDataset:
    """Load and validate a golden dataset JSON file.

    Tolerates an empty ``queries`` list (the placeholder case) so the runner can
    be wired and smoke-tested before the dataset is authored.
    """
    raw = json.loads(Path(path).read_text())
    queries = [Query.from_dict(q) for q in raw.get("queries", [])]
    return GoldenDataset(corpus_dir=raw.get("corpus_dir", ""), queries=queries)
