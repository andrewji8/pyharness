"""Local vector store backed by numpy.

Stores chunks and their embeddings in a compressed ``.npz`` file and performs
cosine-similarity search entirely in-process. No external vector DB required.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from pluggy import HookimplMarker

from pyharness.schema import Chunk, SearchResult
from pyharness.specs import AgentHooks

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")


class LocalVectorStorePlugin:
    """Numpy-backed vector store for local knowledge bases.

    Parameters
    ----------
    store_path:
        Path to the ``.npz`` persistence file. Defaults to
        ``pyharness_vectors.npz`` in the current working directory.
    """

    def __init__(self, store_path: str = "pyharness_vectors.npz") -> None:
        self.store_path = store_path
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    @hookimpl
    async def initialize(self) -> None:
        """Load persisted vectors from disk (best-effort)."""
        if os.path.exists(self.store_path):
            try:
                data = np.load(self.store_path, allow_pickle=True)
                self.vectors = data["vectors"]
                raw_chunks = data["chunks"]
                self.chunks = [Chunk.model_validate(c) for c in raw_chunks]
            except Exception as exc:
                logger.warning("Failed to load vector store: %s", exc)
                self.chunks = []
                self.vectors = None

    @hookimpl
    async def store_chunks(self, chunks: list[Chunk]) -> None:
        """Append chunks (with embeddings) to the store and persist."""
        if not chunks:
            return
        self.chunks.extend(chunks)
        matrix = np.array([c.embedding for c in chunks if c.embedding is not None], dtype=np.float32)
        if matrix.size == 0:
            return
        if self.vectors is None:
            self.vectors = matrix
        else:
            self.vectors = np.vstack([self.vectors, matrix])
        self._save()

    @hookimpl
    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Return the top-k chunks by cosine similarity."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        if q.size == 0:
            return []
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1e-10, norms)
        scores = np.dot(self.vectors, q) / norms

        if filter:
            mask = np.array([self._match_filter(c, filter) for c in self.chunks])
            scores = np.where(mask, scores, -1.0)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[SearchResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=float(scores[idx]),
                    metadata=chunk.metadata,
                )
            )
        return results

    @hookimpl
    async def delete_by_source(self, source: str) -> int:
        """Remove all chunks whose ``source_file`` matches *source*."""
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.metadata.get("source_file") != source]
        if self.chunks:
            self.vectors = np.array([c.embedding for c in self.chunks if c.embedding is not None], dtype=np.float32)
        else:
            self.vectors = None
        self._save()
        return before - len(self.chunks)

    @hookimpl
    async def get_store_stats(self) -> dict[str, Any]:
        dim = int(self.vectors.shape[1]) if self.vectors is not None else 0
        return {
            "total_chunks": len(self.chunks),
            "vector_dim": dim,
            "store_path": self.store_path,
        }

    def _save(self) -> None:
        """Persist chunks and vectors to disk."""
        try:
            np.savez_compressed(
                self.store_path,
                vectors=self.vectors if self.vectors is not None else np.array([]),
                chunks=np.array([c.model_dump(mode="json") for c in self.chunks]),
            )
        except Exception as exc:
            logger.warning("Failed to persist vector store: %s", exc)

    @staticmethod
    def _match_filter(chunk: Chunk, filter: dict) -> bool:
        for key, value in filter.items():
            if chunk.metadata.get(key) != value:
                return False
        return True


__all__ = ["LocalVectorStorePlugin"]
