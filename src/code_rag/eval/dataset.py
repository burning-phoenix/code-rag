"""
Golden dataset: hand-written query → relevant-span pairs.

Relevance is labelled at the *span* level (file + symbol or file + line range),
not the chunk level, so the dataset stays valid when chunking parameters change.
Any retrieved item overlapping a gold span counts as a hit.
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
