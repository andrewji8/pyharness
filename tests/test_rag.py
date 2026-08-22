"""Tests for Phase 5 RAG: text splitting, embeddings, vector store, and tools."""

from __future__ import annotations

import asyncio
import os
import tempfile

import numpy as np
import pytest
from pluggy import HookimplMarker

from pyharness import Harness
from pyharness.context import SessionContext
from pyharness.core import _settle
from pyharness.plugins.embedding import DummyEmbeddingPlugin, OpenAIEmbeddingPlugin
from pyharness.plugins.text_splitter import RecursiveCharacterSplitter
from pyharness.plugins.vector_store import LocalVectorStorePlugin
from pyharness.schema import (
    Chunk,
    HybridSearchInput,
    HybridSearchResult,
    KnowledgeSearchInput,
    SearchResult,
    ToolResultStatus,
)
from pyharness.plugins.tool_knowledge import KnowledgePlugin
from pyharness.plugins.tool_hybrid_search import HybridSearchPlugin
from pyharness.schema import HarnessConfig

hookimpl = HookimplMarker("pyharness")


def _harness(*plugins, auto_load: bool = False) -> Harness:
    h = Harness(config=HarnessConfig(auto_load_entry_points=auto_load))
    for plugin in plugins:
        h.register_plugin(plugin)
    h.initialize()
    return h


# ---------------------------------------------------------------------------
# 1. Text Splitter
# ---------------------------------------------------------------------------
class TestTextSplitter:
    def test_split_respects_chunk_size(self) -> None:
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=20)
        text = "A" * 250
        chunks = splitter.split(text)
        assert len(chunks) > 1
        max_allowed = splitter.chunk_size + splitter.chunk_overlap
        for chunk in chunks:
            assert len(chunk.content) <= max_allowed

    def test_split_overlap(self) -> None:
        splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
        text = "word " * 20
        chunks = splitter.split(text)
        assert len(chunks) > 1
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1].content[-10:]
            curr_head = chunks[i].content[:10]
            assert prev_tail == curr_head

    def test_split_empty(self) -> None:
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=20)
        assert splitter.split("") == []
        assert splitter.split("   ") == []

    def test_split_with_metadata(self) -> None:
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=0)
        text = "hello world " * 20
        chunks = splitter.split(text, metadata={"source_file": "test.py"})
        assert all(c.metadata["source_file"] == "test.py" for c in chunks)
        assert all(c.chunk_id.startswith("test.py_") for c in chunks)


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------
class TestEmbedding:
    def test_dummy_embedding_deterministic(self) -> None:
        plugin = DummyEmbeddingPlugin(dim=64)
        vec1 = asyncio.run(plugin.embed_query("hello world"))
        vec2 = asyncio.run(plugin.embed_query("hello world"))
        assert vec1 == vec2
        assert len(vec1) == 64
        assert all(isinstance(v, float) for v in vec1)

    def test_dummy_embedding_batch(self) -> None:
        plugin = DummyEmbeddingPlugin(dim=32)
        results = asyncio.run(plugin.embed_texts(["a", "b", "c"]))
        assert len(results) == 3
        assert all(len(v) == 32 for v in results)
        assert results[0] != results[1]


# ---------------------------------------------------------------------------
# 3. Vector Store
# ---------------------------------------------------------------------------
class TestVectorStore:
    def test_store_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "vectors.npz")
            store = LocalVectorStorePlugin(store_path=store_path)
            asyncio.run(store.initialize())

            chunks = [
                Chunk(chunk_id="c1", content="Python is great", metadata={"source_file": "a.py"}, embedding=[1.0, 0.0]),
                Chunk(chunk_id="c2", content="Rust is fast", metadata={"source_file": "b.py"}, embedding=[0.0, 1.0]),
                Chunk(chunk_id="c3", content="Python async", metadata={"source_file": "a.py"}, embedding=[0.9, 0.1]),
            ]
            asyncio.run(store.store_chunks(chunks))
            results = asyncio.run(store.search_similar([1.0, 0.0], top_k=2))
            assert len(results) == 2
            assert results[0].chunk_id == "c1"
            assert results[0].score > results[1].score

    def test_delete_by_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "vectors.npz")
            store = LocalVectorStorePlugin(store_path=store_path)
            asyncio.run(store.initialize())
            chunks = [
                Chunk(chunk_id="c1", content="text", metadata={"source_file": "a.py"}, embedding=[1.0]),
                Chunk(chunk_id="c2", content="text2", metadata={"source_file": "b.py"}, embedding=[0.5]),
            ]
            asyncio.run(store.store_chunks(chunks))
            removed = asyncio.run(store.delete_by_source("a.py"))
            assert removed == 1
            stats = asyncio.run(store.get_store_stats())
            assert stats["total_chunks"] == 1

    def test_get_store_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "vectors.npz")
            store = LocalVectorStorePlugin(store_path=store_path)
            asyncio.run(store.initialize())
            stats = asyncio.run(store.get_store_stats())
            assert stats["total_chunks"] == 0
            assert stats["vector_dim"] == 0


# ---------------------------------------------------------------------------
# 4. Knowledge Search (end-to-end)
# ---------------------------------------------------------------------------
class TestKnowledgeSearch:
    def test_knowledge_search_returns_results(self) -> None:
        store = LocalVectorStorePlugin()
        h = _harness(store)
        asyncio.run(store.initialize())

        chunks = [
            Chunk(chunk_id="k1", content="Python async tutorial", metadata={"source_file": "py.md"}, embedding=[1.0, 0.0]),
            Chunk(chunk_id="k2", content="Rust ownership guide", metadata={"source_file": "rs.md"}, embedding=[0.0, 1.0]),
        ]
        asyncio.run(store.store_chunks(chunks))

        # Create a fake embedding plugin that returns the chunks' own embeddings
        class FakeEmbedding:
            @hookimpl
            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] if any("Python" in t or "async" in t for t in texts) else [0.0, 1.0] for _ in texts]

            @hookimpl
            async def embed_query(self, query: str) -> list[float]:
                return [1.0, 0.0] if "Python" in query else [0.0, 1.0]

        h.bus.register(FakeEmbedding())

        plugin = KnowledgePlugin()
        plugin.harness = h
        result = asyncio.run(plugin._knowledge_search({"query": "Python", "top_k": 2}))
        assert result.status == ToolResultStatus.OK
        assert "Python async tutorial" in result.output["result"]

    def test_knowledge_search_empty(self) -> None:
        store = LocalVectorStorePlugin()
        h = _harness(store)
        asyncio.run(store.initialize())

        plugin = KnowledgePlugin()
        plugin.harness = h
        result = asyncio.run(plugin._knowledge_search({"query": "nothing", "top_k": 5}))
        assert result.status == ToolResultStatus.OK
        assert "未找到" in result.output["result"]


# ---------------------------------------------------------------------------
# 5. Ingest Directory
# ---------------------------------------------------------------------------
class TestIngestDirectory:
    def test_ingest_directory_txt_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.txt"), "w") as f:
                f.write("Python is great. " * 100)
            with open(os.path.join(tmpdir, "b.txt"), "w") as f:
                f.write("Rust is fast. " * 100)

            embedding = DummyEmbeddingPlugin()
            store = LocalVectorStorePlugin()
            h = _harness(embedding, store)
            asyncio.run(store.initialize())

            plugin = KnowledgePlugin()
            plugin.harness = h
            result = asyncio.run(
                plugin._ingest_directory(
                    {"path": tmpdir, "patterns": ["*.txt"], "chunk_size": 200, "chunk_overlap": 50}
                )
            )
            assert result.status == ToolResultStatus.OK
            assert "2 个文件" in result.output["result"]
            assert "切片" in result.output["result"]

    def test_ingest_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding = DummyEmbeddingPlugin()
            store = LocalVectorStorePlugin()
            h = _harness(embedding, store)
            asyncio.run(store.initialize())

            plugin = KnowledgePlugin()
            plugin.harness = h
            result = asyncio.run(plugin._ingest_directory({"path": tmpdir, "patterns": ["*.txt"]}))
            assert result.status == ToolResultStatus.OK
            assert "未找到" in result.output["result"]


# ---------------------------------------------------------------------------
# 6. Hybrid Search
# ---------------------------------------------------------------------------
class TestHybridSearch:
    def test_hybrid_search_rrf(self) -> None:
        store = LocalVectorStorePlugin()
        h = _harness(store)
        asyncio.run(store.initialize())

        chunks = [
            Chunk(chunk_id="h1", content="Python async programming", metadata={"source_file": "py.md"}, embedding=[1.0, 0.0]),
            Chunk(chunk_id="h2", content="Rust async runtime", metadata={"source_file": "rs.md"}, embedding=[0.0, 1.0]),
        ]
        asyncio.run(store.store_chunks(chunks))

        class FakeEmbedding:
            @hookimpl
            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

            @hookimpl
            async def embed_query(self, query: str) -> list[float]:
                return [1.0, 0.0]

        h.bus.register(FakeEmbedding())

        plugin = HybridSearchPlugin()
        plugin.harness = h
        result = asyncio.run(
            plugin._hybrid_search({"query": "Python async", "top_k": 2, "fts_weight": 0.3, "vector_weight": 0.7})
        )
        assert result.status == ToolResultStatus.OK
        assert "混合分数" in result.output["result"]


__all__ = [
    "TestTextSplitter",
    "TestEmbedding",
    "TestVectorStore",
    "TestKnowledgeSearch",
    "TestIngestDirectory",
    "TestHybridSearch",
]
