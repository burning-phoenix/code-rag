"""
Tests for the batched enrichment pipeline.

Validates parsing reliability against mock LLM responses under varied
conditions WITHOUT hitting the real API. Ensures the soul+constitution
prompt produces parseable output and the fallback logic works correctly.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_rag.chunker import Chunk
from code_rag.enrichment import (
    ENRICHMENT_SYSTEM_PROMPT,
    _build_batch_prompt,
    _parse_batch_response,
    enrich_chunks,
    get_embedding_text,
)


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
        items.append({
            "id": i,
            "summary": f"This is chunk {i} which does something useful.",
            "hypothetical_questions": [
                f"What does chunk {i} do?",
                f"When would I use chunk {i}?",
            ],
        })
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
        content = json.dumps({
            "chunks": [
                {"id": 1, "summary": "Does something."}
            ]
        })
        results, error = _parse_batch_response(content, 1)
        assert results is None
        assert error is not None

    def test_parse_empty_summary(self):
        """Validator rejects empty string summary."""
        content = json.dumps({
            "chunks": [
                {
                    "id": 1,
                    "summary": "   ",
                    "hypothetical_questions": ["Q1?"],
                }
            ]
        })
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

    def test_positional_fallback(self):
        """Wrong IDs but correct count uses positional assignment."""
        items = [
            {"id": 10, "summary": "A.", "hypothetical_questions": ["Q1?"]},
            {"id": 20, "summary": "B.", "hypothetical_questions": ["Q2?"]},
        ]
        content = json.dumps({"chunks": items})
        results, error = _parse_batch_response(content, 2)
        assert error is None
        assert results is not None
        assert len(results) == 2
        assert results[0].summary == "A."
        assert results[1].summary == "B."

    def test_extra_whitespace_in_summary(self):
        """Summary with leading/trailing whitespace is stripped."""
        content = json.dumps({
            "chunks": [
                {
                    "id": 1,
                    "summary": "  Does something useful.  ",
                    "hypothetical_questions": ["Q?"],
                }
            ]
        })
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
            _make_chunk(text="void main() {}", file_name="b.glsl", language="glsl", chunk_type="code"),
            _make_chunk(text="# Header\nSome docs", file_name="c.md", language=None, chunk_type="markdown"),
            _make_chunk(text="plain text here", file_name="d.txt", language=None, chunk_type="plaintext"),
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
# End-to-end mock test
# ---------------------------------------------------------------------------


def _mock_openrouter_response(n: int):
    """Create a mock httpx response with valid enrichment for n chunks."""
    body = _make_valid_response(n)
    # Use MagicMock (not AsyncMock) because httpx response.json() is sync
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": body}}]
    }
    return mock_response


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_enrich_chunks_batched(self):
        """Full pipeline with mocked HTTP — 7 chunks in batches of 3."""
        chunks = [
            _make_chunk(text=f"def func_{i}():\n    pass\n    # line 3\n", start_line=1, end_line=4)
            for i in range(7)
        ]

        # Mock will be called 3 times: batch(3) + batch(3) + batch(1)
        call_count = 0

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            # Determine batch size from the prompt
            messages = kwargs.get("json", {}).get("messages", [])
            user_msg = messages[-1]["content"] if messages else ""
            # Count chunk delimiters
            n = user_msg.count("--- CHUNK ")
            return _mock_openrouter_response(n)

        with patch("code_rag.enrichment.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await enrich_chunks(
                chunks, api_key="test-key", model="test-model", batch_size=3,
            )

        assert len(result) == 7
        # All chunks should be enriched
        for chunk in result:
            assert chunk.summary is not None
            assert chunk.hypothetical_questions is not None
            assert len(chunk.hypothetical_questions) >= 1

        # Should have made 3 batch calls (3+3+1)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_skips_short_chunks(self):
        """Chunks with < 3 lines are skipped."""
        chunks = [
            _make_chunk(text="short", start_line=1, end_line=2),  # 2 lines — skip
            _make_chunk(text="def f():\n    pass\n    # 3\n", start_line=1, end_line=4),  # 4 lines — enrich
        ]

        async def mock_post(url, **kwargs):
            return _mock_openrouter_response(1)

        with patch("code_rag.enrichment.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await enrich_chunks(
                chunks, api_key="test-key", model="test-model", batch_size=5,
            )

        assert result[0].summary is None  # skipped
        assert result[1].summary is not None  # enriched

    @pytest.mark.asyncio
    async def test_no_api_key_skips(self):
        """Empty API key skips enrichment entirely."""
        chunks = [_make_chunk(start_line=1, end_line=10)]
        result = await enrich_chunks(chunks, api_key="", model="test-model")
        assert result[0].summary is None
