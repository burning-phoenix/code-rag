"""
OpenRouter-backed providers: embeddings and chat completions.

These own all HTTP transport, retry, and rate-limit handling for OpenRouter's
OpenAI-compatible API. Higher layers depend only on the
:class:`~code_rag.protocols.EmbeddingProvider` / ``LLMProvider`` contracts.
"""

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
EMBEDDING_DIM = 4096  # Confirmed from HuggingFace: supports 32–4096


class OpenRouterEmbeddings:
    """Embedding provider backed by OpenRouter (Qwen3-Embedding-8B by default)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = EMBEDDING_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        dimension: int = EMBEDDING_DIM,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model = model
        self._base_url = base_url
        self._dimension = dimension
        self._max_retries = max_retries

    @property
    def dimension(self) -> int:
        """Vector dimensionality this provider produces (the collection size)."""
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via OpenRouter, preserving input order.

        Raises:
            ValueError: if no API key is configured.
            httpx.HTTPStatusError: on non-retryable API errors.
        """
        if not self._api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Add it to your project's .env file "
                "or pass api_key argument."
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }
        timeout = 30.0 + 10.0 * len(texts)

        response = None
        for attempt in range(1 + self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/embeddings",
                        headers=headers,
                        json=payload,
                    )

                    if response.status_code == 429:
                        wait = 2**attempt
                        logger.warning("Rate limited (429), waiting %ds before retry", wait)
                        await asyncio.sleep(wait)
                        continue

                    response.raise_for_status()

            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                logger.warning(
                    "Embedding call failed (attempt %d/%d): %s",
                    attempt + 1,
                    1 + self._max_retries,
                    e,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                raise

            break

        assert response is not None  # loop either breaks with a response or raises
        data = response.json()
        # Sort by index to ensure correct ordering
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in embeddings_data]

        logger.info(
            "Embedded %d texts → %d-dim vectors (model: %s)",
            len(texts),
            len(embeddings[0]) if embeddings else 0,
            self._model,
        )
        return embeddings


class OpenRouterLLM:
    """Chat-completion provider backed by OpenRouter.

    Owns HTTP transport and rate-limit retry only. Prompt construction, response
    parsing, and parse-failure retries live in :mod:`code_rag.enrichment`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._max_retries = max_retries

    async def complete(self, messages: list[dict], model: str, **kwargs: Any) -> str:
        """Run a chat completion and return the assistant message content.

        Keyword args (``max_tokens``, ``temperature``, ``response_format``,
        ``timeout``) are forwarded to the API. Retries transient failures and
        429s with exponential backoff; raises after exhausting retries.
        """
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY not set.")

        timeout = kwargs.pop("timeout", 120.0)
        body = {"model": model, "messages": messages, **kwargs}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1 + self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=body,
                    )

                    if response.status_code == 429:
                        wait = 2**attempt
                        logger.warning("Rate limited, waiting %ds before retry", wait)
                        await asyncio.sleep(wait)
                        continue

                    response.raise_for_status()
                    content: str = response.json()["choices"][0]["message"]["content"]
                    return content

            except Exception as e:  # noqa: BLE001 — retry any transport error, then surface it
                last_error = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    1 + self._max_retries,
                    e,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)

        raise RuntimeError(f"LLM completion failed after retries: {last_error}")
