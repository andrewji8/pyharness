"""Text splitting utilities for RAG.

Provides ``RecursiveCharacterSplitter``: a LangChain-compatible recursive
character splitter that tries multiple separators in priority order.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pyharness.schema import Chunk

logger = logging.getLogger(__name__)


class RecursiveCharacterSplitter:
    """Recursive character text splitter.

    Tries to split on each separator in order, falling back to the next
    separator when a chunk exceeds ``chunk_size``. Finally falls back to
    character-level splitting.

    Parameters
    ----------
    chunk_size:
        Maximum chunk size in characters.
    chunk_overlap:
        Number of overlapping characters between adjacent chunks.
    separators:
        Ordered list of separators to try. Defaults to paragraph, line,
        sentence, and character boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* into chunks and return :class:`Chunk` objects."""
        if not text.strip():
            return []

        raw_chunks = self._split_recursive(text, self.separators)
        overlapped = self._apply_overlap(raw_chunks)
        chunks: list[Chunk] = []
        source = metadata.get("source_file", "unknown") if metadata else "unknown"
        for i, part in enumerate(overlapped):
            chunk_meta = dict(metadata) if metadata else {}
            chunk_meta.update(
                {
                    "source_file": source,
                    "chunk_index": i,
                    "char_start": sum(len(p) for p in overlapped[:i]),
                    "char_end": sum(len(p) for p in overlapped[: i + 1]),
                }
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{source}_{i}",
                    content=part,
                    metadata=chunk_meta,
                )
            )
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the given separators."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = separators[0]
        remaining = separators[1:] if len(separators) > 1 else [""]

        parts = text.split(separator) if separator else list(text)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + separator + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split_recursive(part, remaining))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between adjacent chunks."""
        if len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = (
                chunks[i - 1][-self.chunk_overlap :]
                if len(chunks[i - 1]) > self.chunk_overlap
                else chunks[i - 1]
            )
            overlapped.append(prev_tail + chunks[i])
        return overlapped


def read_file_safe(path: str, max_bytes: int = 1_048_576) -> str | None:
    """Read a text file safely, returning ``None`` on failure or if it exceeds *max_bytes*."""
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            logger.warning("Skipping large file (%d bytes): %s", size, path)
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


__all__ = ["RecursiveCharacterSplitter", "read_file_safe"]
