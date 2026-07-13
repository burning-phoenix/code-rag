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

# Embedding provider routing, fastest first. Measured 2026-07-10 (10-text
# batch): SiliconFlow 1.6s, Nebius 9.7s, DeepInfra 41s. Only the listed
# providers are allowed (allow_fallbacks=false), so the badly degraded one is
# never used and routing can't silently wander mid-run — embeddings within a
# run stay mutually comparable (SiliconFlow's deployment is fp8-quantized, so
# its vectors differ minutely from Nebius's; Nebius remains the in-list backup
# if SiliconFlow errors). Pass provider_order=() to restore default routing.
EMBEDDING_PROVIDER_ORDER: tuple[str, ...] = ("siliconflow", "nebius")

# Chat-completion routing preference, fastest first. A *soft* preference — see
# the comment at the payload site in OpenRouterLLM.complete. Measured
# 2026-07-10: llama-4-scout — Groq ~280 tok/s, Google Vertex ~167, DeepInfra/
# Novita ~35; gemma-4-31b (not on Groq/Vertex) — SambaNova ~50 tok/s, Novita
# ~14, Together ~8, SiliconFlow ~3. Order entries only apply where the model
# is hosted, so one list serves both.
LLM_PROVIDER_ORDER: tuple[str, ...] = ("groq", "google-vertex", "sambanova")


class OpenRouterEmbeddings:
    """Embedding provider backed by OpenRouter (Qwen3-Embedding-8B by default)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = EMBEDDING_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        dimension: int = EMBEDDING_DIM,
        max_retries: int = 2,
        provider_order: tuple[str, ...] = EMBEDDING_PROVIDER_ORDER,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model = model
        self._base_url = base_url
        self._dimension = dimension
        self._max_retries = max_retries
        self._provider_order = provider_order

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
        payload: dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }
        if self._provider_order:
            payload["provider"] = {
                "order": list(self._provider_order),
                "allow_fallbacks": False,
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
        # OpenRouter can report provider errors as HTTP 200 with an error body
        # (e.g. input over the model's token limit) — surface the message instead
        # of dying on a KeyError.
        if "data" not in data:
            raise RuntimeError(
                f"Embeddings response has no 'data' (provider error?): {str(data)[:500]}"
            )
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
        provider_order: tuple[str, ...] = LLM_PROVIDER_ORDER,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._max_retries = max_retries
        self._provider_order = provider_order

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
        if self._provider_order:
            # Soft preference (fallbacks stay allowed): unlike embeddings, the
            # served model is caller-chosen, so a hard pin would break any model
            # the preferred providers don't host. Output is text validated by
            # the enrichment parser — provider mixing carries no consistency risk.
            body["provider"] = {"order": list(self._provider_order)}
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
                    data = response.json()
                    # OpenRouter can report provider errors as HTTP 200 with an
                    # error body — surface the message instead of a KeyError.
                    if "choices" not in data:
                        raise RuntimeError(
                            f"Completion response has no 'choices' (provider error?): "
                            f"{str(data)[:500]}"
                        )
                    content: str = data["choices"][0]["message"]["content"]
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
