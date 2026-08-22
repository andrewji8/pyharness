"""Embedding providers for PyHarness RAG.

Provides:
- ``OpenAIEmbeddingPlugin``: OpenAI-compatible embedding API (supports OpenAI,
  DeepSeek, 通义等).
- ``DummyEmbeddingPlugin``: deterministic hash-based embedding for offline tests.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from pluggy import HookimplMarker

from pyharness.specs import AgentHooks

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")


class OpenAIEmbeddingPlugin:
    """OpenAI-compatible embedding service.

    Reads ``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY`` and ``OPENAI_BASE_URL`` /
    ``EMBEDDING_BASE_URL`` from the environment.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv(
            "EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model
        self._dimension: int | None = None

    @hookimpl
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed up to 100 texts per request with retry."""
        import asyncio
        import httpx

        if not texts:
            return []

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        results: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload: dict[str, Any] = {"model": self.model, "input": batch}
            for attempt in range(4):
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            f"{self.base_url}/embeddings",
                            headers=headers,
                            json=payload,
                        )
                        response.raise_for_status()
                        data = response.json()
                        sorted_data = sorted(data.get("data", []), key=lambda x: x["index"])
                        results.extend(item["embedding"] for item in sorted_data)
                        if self._dimension is None and results:
                            self._dimension = len(results[0])
                        break
                except Exception as exc:
                    if attempt == 3:
                        logger.error("embed_texts failed after retries: %s", exc)
                        raise
                    await asyncio.sleep(2 ** attempt)
        return results

    @hookimpl
    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self.embed_texts([query])
        return result[0] if result else []


class DummyEmbeddingPlugin:
    """Deterministic hash-based embedding for offline testing."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    @hookimpl
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    @hookimpl
    async def embed_query(self, query: str) -> list[float]:
        return self._hash_to_vector(query)

    def _hash_to_vector(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).hexdigest()
        repeated = (h * ((self.dim // len(h)) + 1))[: self.dim * 2]
        if len(repeated) < self.dim * 2:
            repeated = repeated.ljust(self.dim * 2, "0")
        vec = [int(repeated[i : i + 2], 16) / 255.0 for i in range(0, self.dim * 2, 2)]
        return vec


__all__ = ["OpenAIEmbeddingPlugin", "DummyEmbeddingPlugin"]
