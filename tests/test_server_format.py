"""Tests for the MCP result formatter (pure function, no providers needed)."""

from code_rag.server import _format_result


def _result(**overrides) -> dict:
    base = dict(
        text="some content",
        score=0.9876,
        file_name="a.py",
        start_line=1,
        end_line=10,
        language=None,
        symbol_name=None,
        symbol_type=None,
        summary=None,
        math_density=0.0,
    )
    base.update(overrides)
    return base


def test_code_chunk_is_wrapped_in_code_fence():
    out = _format_result(_result(language="python", text="def f(): ..."), 1)
    assert "```python" in out
    assert "def f(): ..." in out


def test_markdown_chunk_is_plain_text():
    out = _format_result(_result(language=None, text="just prose"), 1)
    assert "```" not in out
    assert "just prose" in out


def test_long_text_is_truncated():
    out = _format_result(_result(text="x" * 5000), 1, max_chars=100)
    assert "truncated" in out
    assert "5000 chars total" in out


def test_summary_and_math_density_included_when_present():
    out = _format_result(_result(summary="does a thing", math_density=0.42), 1)
    assert "**Summary:** does a thing" in out
    assert "**Math density:** 0.42" in out


def test_summary_and_math_density_omitted_when_absent():
    out = _format_result(_result(summary=None, math_density=0.0), 1)
    assert "Summary:" not in out
    assert "Math density:" not in out
