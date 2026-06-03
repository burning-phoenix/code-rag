"""
Tests for the batched enrichment pipeline.

Validates parsing reliability against mock LLM responses under varied
conditions WITHOUT hitting the real API. Ensures the soul+constitution
prompt produces parseable output and the fallback logic works correctly.
"""

import json

import pytest

from code_rag.enrichment import (
    ENRICHMENT_SYSTEM_PROMPT,
    _build_batch_prompt,
    _parse_batch_response,
    enrich_chunks,
    get_embedding_text,
)
from code_rag.models import Chunk
from code_rag.providers import FakeLLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str = "def foo():\n    return 42\n",
    file_name: str = "test.py",
    chunk_type: str = "code",
    language: str | None = "python",
    start_line: int = 1,
    end_line: int = 10,
) -> Chunk:
    return Chunk(
        text=text,
        file_name=file_name,
        section_title="foo",
        start_line=start_line,
        end_line=end_line,
        chunk_type=chunk_type,
        language=language,
    )


def _make_valid_response(n: int) -> str:
    """Build a well-formed JSON response for n chunks."""
    items = []
    for i in range(1, n + 1):
        items.append(
            {
                "id": i,
                "summary": f"This is chunk {i} which does something useful.",
                "hypothetical_questions": [
                    f"What does chunk {i} do?",
                    f"When would I use chunk {i}?",
                ],
            }
        )
    return json.dumps({"chunks": items})


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


class TestParseBatchResponse:
    def test_parse_perfect_json(self):
        """Well-formed 5-chunk JSON response parses correctly."""
        content = _make_valid_response(5)
        results, error = _parse_batch_response(content, 5)
        assert error is None
        assert results is not None
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.id == i + 1
            assert "chunk" in r.summary.lower()
            assert len(r.hypothetical_questions) == 2

    def test_parse_json_in_markdown_fences(self):
        """Regex fallback extracts JSON from markdown code fences."""
        inner = _make_valid_response(3)
        content = f"```json\n{inner}\n```"
        results, error = _parse_batch_response(content, 3)
        assert error is None
        assert results is not None
        assert len(results) == 3

    def test_parse_json_with_preamble(self):
        """Regex fallback handles text before/after JSON."""
        inner = _make_valid_response(2)
        content = f"Here is the analysis:\n{inner}\n\nHope this helps!"
        results, error = _parse_batch_response(content, 2)
        assert error is None
        assert results is not None
        assert len(results) == 2

    def test_parse_wrong_chunk_count(self):
        """Returns error when response has fewer entries than expected."""
        content = _make_valid_response(4)
        results, error = _parse_batch_response(content, 5)
        assert results is None
        assert error is not None
        assert "Expected 5" in error

    def test_parse_missing_fields(self):
        """Pydantic catches missing hypothetical_questions key."""
        content = json.dumps({"chunks": [{"id": 1, "summary": "Does something."}]})
        results, error = _parse_batch_response(content, 1)
        assert results is None
        assert error is not None

    def test_parse_empty_summary(self):
        """Validator rejects empty string summary."""
        content = json.dumps(
            {
                "chunks": [
                    {
                        "id": 1,
                        "summary": "   ",
                        "hypothetical_questions": ["Q1?"],
                    }
                ]
            }
        )
        results, error = _parse_batch_response(content, 1)
        assert results is None
        assert error is not None

    def test_parse_garbled_response(self):
        """Graceful failure with error message on garbage input."""
        content = "This is not JSON at all! Random text here 🤖"
        results, error = _parse_batch_response(content, 3)
        assert results is None
        assert error is not None
        assert "Could not parse" in error

    def test_id_reordering(self):
        """Out-of-order IDs are sorted correctly."""
        items = [
            {"id": 3, "summary": "Third.", "hypothetical_questions": ["Q3?"]},
            {"id": 1, "summary": "First.", "hypothetical_questions": ["Q1?"]},
            {"id": 2, "summary": "Second.", "hypothetical_questions": ["Q2?"]},
        ]
        content = json.dumps({"chunks": items})
        results, error = _parse_batch_response(content, 3)
        assert error is None
        assert results is not None
        assert [r.id for r in results] == [1, 2, 3]
        assert results[0].summary == "First."

    def test_garbled_ids_are_a_parse_failure(self):
        """Right count but ids != 1..n fails rather than guessing by position.

        Trusting positional order would silently attach the wrong summary to the
        wrong chunk. We instead signal failure so the caller retries / falls back
        to per-chunk calls where ids are trivial.
        """
        items = [
            {"id": 10, "summary": "A.", "hypothetical_questions": ["Q1?"]},
            {"id": 20, "summary": "B.", "hypothetical_questions": ["Q2?"]},
        ]
        content = json.dumps({"chunks": items})
        results, error = _parse_batch_response(content, 2)
        assert results is None
        assert error is not None
        assert "ids" in error.lower()

    def test_extra_whitespace_in_summary(self):
        """Summary with leading/trailing whitespace is stripped."""
        content = json.dumps(
            {
                "chunks": [
                    {
                        "id": 1,
                        "summary": "  Does something useful.  ",
                        "hypothetical_questions": ["Q?"],
                    }
                ]
            }
        )
        results, error = _parse_batch_response(content, 1)
        assert error is None
        assert results is not None
        assert results[0].summary == "Does something useful."


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_batch_prompt_structure(self):
        """5 varied chunks produce correct delimiters and metadata."""
        chunks = [
            _make_chunk(text="def foo(): pass", file_name="a.py", language="python"),
            _make_chunk(
                text="void main() {}", file_name="b.glsl", language="glsl", chunk_type="code"
            ),
            _make_chunk(
                text="# Header\nSome docs", file_name="c.md", language=None, chunk_type="markdown"
            ),
            _make_chunk(
                text="plain text here", file_name="d.txt", language=None, chunk_type="plaintext"
            ),
            _make_chunk(text="fn main() {}", file_name="e.rs", language="rust"),
        ]
        prompt = _build_batch_prompt(chunks)

        # All chunk delimiters present
        for i in range(1, 6):
            assert f"--- CHUNK {i} ---" in prompt

        # File names present
        for name in ["a.py", "b.glsl", "c.md", "d.txt", "e.rs"]:
            assert name in prompt

        # Language notes for code chunks
        assert "(python)" in prompt
        assert "(glsl)" in prompt
        assert "(rust)" in prompt

        # No language note for markdown/plaintext
        assert prompt.count("(python)") == 1

    def test_text_truncation(self):
        """Chunk text exceeding MAX_CHUNK_TEXT is truncated."""
        long_text = "x" * 5000
        chunks = [_make_chunk(text=long_text)]
        prompt = _build_batch_prompt(chunks)

        # The prompt should not contain the full 5000 chars
        assert len(prompt) < 5000 + 500  # prompt overhead

    def test_single_chunk_prompt(self):
        """Single-chunk batch still produces valid prompt."""
        chunks = [_make_chunk()]
        prompt = _build_batch_prompt(chunks)
        assert "--- CHUNK 1 ---" in prompt
        assert "following 1 code/documentation" in prompt


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_contains_identity(self):
        """System prompt contains the soul identity."""
        assert "CodeSage" in ENRICHMENT_SYSTEM_PROMPT

    def test_contains_constitution_keywords(self):
        """System prompt contains key constitution terms."""
        for keyword in ["JSON", "chunks", "summary", "hypothetical_questions", "CONSTITUTION"]:
            assert keyword in ENRICHMENT_SYSTEM_PROMPT

    def test_contains_all_rules(self):
        """System prompt has all 6 constitution rules."""
        for i in range(1, 7):
            assert f"{i}." in ENRICHMENT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# get_embedding_text tests
# ---------------------------------------------------------------------------


class TestGetEmbeddingText:
    def test_no_enrichment(self):
        """Returns just the raw text when no enrichment."""
        chunk = _make_chunk(text="hello world")
        assert get_embedding_text(chunk) == "hello world"

    def test_with_summary(self):
        """Includes summary when present."""
        chunk = _make_chunk(text="hello")
        chunk.summary = "Greets the world."
        result = get_embedding_text(chunk)
        assert "hello" in result
        assert "Summary: Greets the world." in result

    def test_with_questions(self):
        """Includes questions when present."""
        chunk = _make_chunk(text="hello")
        chunk.hypothetical_questions = ["What does this do?", "When to use it?"]
        result = get_embedding_text(chunk)
        assert "Questions this answers:" in result
        assert "What does this do?" in result

    def test_with_both(self):
        """Includes both summary and questions."""
        chunk = _make_chunk(text="hello")
        chunk.summary = "Greets."
        chunk.hypothetical_questions = ["Q1?"]
        result = get_embedding_text(chunk)
        assert "Summary: Greets." in result
        assert "Q1?" in result


# ---------------------------------------------------------------------------
# enrich_chunks orchestration (with FakeLLM — no network)
# ---------------------------------------------------------------------------


class TestEnrichChunks:
    @pytest.mark.asyncio
    async def test_enrich_chunks_batched(self):
        """7 chunks in batches of 3 are all enriched via the LLM provider."""
        chunks = [
            _make_chunk(text=f"def func_{i}():\n    pass\n    # line 3\n", start_line=1, end_line=4)
            for i in range(7)
        ]
        llm = FakeLLM()

        result = await enrich_chunks(chunks, llm, model="test-model", batch_size=3)

        assert len(result) == 7
        for chunk in result:
            assert chunk.summary is not None
            assert chunk.hypothetical_questions is not None
            assert len(chunk.hypothetical_questions) >= 1

        # Three batch calls: 3 + 3 + 1
        assert len(llm.calls) == 3

    @pytest.mark.asyncio
    async def test_skips_short_chunks(self):
        """Chunks with < 3 lines are not sent for enrichment."""
        chunks = [
            _make_chunk(text="short", start_line=1, end_line=2),  # 2 lines — skip
            _make_chunk(text="def f():\n    pass\n    # 3\n", start_line=1, end_line=4),  # enrich
        ]
        llm = FakeLLM()

        result = await enrich_chunks(chunks, llm, model="test-model", batch_size=5)

        assert result[0].summary is None  # skipped
        assert result[1].summary is not None  # enriched

    @pytest.mark.asyncio
    async def test_unparseable_response_leaves_chunks_unenriched(self):
        """A response that can never be parsed leaves chunks un-enriched (graceful)."""
        chunks = [_make_chunk(text="def f():\n    pass\n    # 3\n", start_line=1, end_line=4)]
        llm = FakeLLM(response="this is not JSON at all")

        result = await enrich_chunks(chunks, llm, model="test-model")

        assert result[0].summary is None

    @pytest.mark.asyncio
    async def test_garbled_batch_ids_fall_back_to_per_chunk(self):
        """A batch with garbled ids must not misattribute; it falls back to singles.

        The fake returns the right count but wrong ids for multi-chunk batches
        (which must be rejected) and valid ids for single-chunk calls. Each chunk
        should end up with the single-call summary, never a positionally-guessed
        batch summary.
        """

        class GarbledBatchLLM:
            def __init__(self) -> None:
                self.calls: list[list[dict]] = []

            async def complete(self, messages: list[dict], model: str, **kwargs) -> str:
                self.calls.append(messages)
                n = messages[-1]["content"].count("--- CHUNK ")
                if n <= 1:
                    items = [{"id": 1, "summary": "from-single", "hypothetical_questions": ["q?"]}]
                else:
                    # Correct count, garbled ids — must be treated as a parse failure.
                    items = [
                        {"id": 100 + i, "summary": "from-batch", "hypothetical_questions": ["q?"]}
                        for i in range(n)
                    ]
                return json.dumps({"chunks": items})

        chunks = [
            _make_chunk(text=f"def f{i}():\n    pass\n    # 3\n", start_line=1, end_line=4)
            for i in range(2)
        ]
        llm = GarbledBatchLLM()

        result = await enrich_chunks(chunks, llm, model="test-model", batch_size=2)

        assert all(c.summary == "from-single" for c in result)
        assert all(c.summary != "from-batch" for c in result)
