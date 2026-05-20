"""
Embedding client: wraps OpenRouter's OpenAI-compatible embeddings API
for the Qwen3-Embedding-8B model.
"""

import asyncio
import os
import logging

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
EMBEDDING_DIM = 4096  # Confirmed from HuggingFace: supports 32–4096


async def embed_texts(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    api_key: str | None = None,
    max_retries: int = 2,
    timeout: float | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts via OpenRouter.

    Args:
        texts: List of text strings to embed.
        model: Embedding model identifier.
        api_key: OpenRouter API key (falls back to env var).
        max_retries: Number of retry attempts on transient failures.
        timeout: HTTP timeout in seconds (default: scales with batch size).

    Returns:
        List of embedding vectors (list of floats), one per input text.

    Raises:
        httpx.HTTPStatusError: On non-retryable API errors.
        ValueError: On missing API key.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Add it to your project's .env file "
            "or pass api_key argument."
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }

    effective_timeout = timeout if timeout is not None else 30.0 + 10.0 * len(texts)

    for attempt in range(1 + max_retries):
        try:
            async with httpx.AsyncClient(timeout=effective_timeout) as client:
                response = await client.post(
                    f"{OPENROUTER_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited (429), waiting %ds before retry", wait)
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()

        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            logger.warning(
                "Embedding call failed (attempt %d/%d): %s",
                attempt + 1, 1 + max_retries, e,
            )
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            raise

        break

    data = response.json()
    # Sort by index to ensure correct ordering
    embeddings_data = sorted(data["data"], key=lambda x: x["index"])
    embeddings = [item["embedding"] for item in embeddings_data]

    logger.info(
        "Embedded %d texts → %d-dim vectors (model: %s)",
        len(texts),
        len(embeddings[0]) if embeddings else 0,
        model,
    )

    return embeddings


async def embed_single(text: str, **kwargs) -> list[float]:
    """Convenience wrapper for embedding a single text."""
    results = await embed_texts([text], **kwargs)
    return results[0]
