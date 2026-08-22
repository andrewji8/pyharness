"""Knowledge tools for PyHarness RAG.

Registers two tools:
- ``knowledge_search``: semantic search over the local vector store.
- ``ingest_directory``: ingest a directory of files into the knowledge base.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
from typing import Any

from pluggy import HookimplMarker

from pyharness.core import _settle
from pyharness.plugins.text_splitter import RecursiveCharacterSplitter, read_file_safe
from pyharness.schema import Chunk, ToolArg, ToolResult, ToolResultStatus, ToolSpec
from pyharness.specs import AgentHooks

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")


class KnowledgePlugin:
    """Tool provider for RAG knowledge base operations."""

    def __init__(self) -> None:
        self.harness: Any = None

    @hookimpl
    def harness_initialized(self, harness: Any) -> None:
        self.harness = harness

    # -- Tool specs -------------------------------------------------------- #
    def _knowledge_search_spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge_search",
            description=(
                "在本地知识库中执行语义搜索。"
                "返回与查询最相关的文本片段及其相似度分数。"
                "适用于查找代码片段、回忆文档内容、检索项目知识。"
                "与 memory_search（FTS5 关键词搜索）互补使用效果更佳。"
            ),
            parameters=(
                ToolArg(name="query", type="string", description="要检索的问题或关键词。", required=True),
                ToolArg(name="top_k", type="integer", description="返回结果数量（默认 5，最大 20）。", required=False),
                ToolArg(
                    name="source_filter",
                    type="string",
                    description="按文件路径过滤（可选）。",
                    required=False,
                ),
            ),
            timeout_seconds=30.0,
        )

    def _ingest_directory_spec(self) -> ToolSpec:
        return ToolSpec(
            name="ingest_directory",
            description=(
                "将本地目录中的文本文件切片、向量化并存入知识库。"
                "支持 .py / .md / .txt 等文本文件。"
                "自动跳过二进制文件和超过 1MB 的文件。"
                "如果目录已摄入过，会先删除旧切片再重新摄入。"
            ),
            parameters=(
                ToolArg(name="path", type="string", description="要摄入的目录路径（相对于 workspace）。", required=True),
                ToolArg(
                    name="patterns",
                    type="array",
                    description="文件匹配模式列表（默认 ['*.py', '*.md', '*.txt']）。",
                    required=False,
                ),
                ToolArg(name="chunk_size", type="integer", description="切片大小（字符数，默认 1000）。", required=False),
                ToolArg(name="chunk_overlap", type="integer", description="切片重叠（字符数，默认 200）。", required=False),
            ),
            timeout_seconds=120.0,
        )

    def _get_store_stats_spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge_stats",
            description="获取当前知识库的统计信息（总切片数、向量维度等）。",
            parameters=(),
            timeout_seconds=10.0,
        )

    # -- Hookimpls --------------------------------------------------------- #
    @hookimpl
    def get_tool_specs(self, context: Any) -> tuple[ToolSpec, ...]:
        return (
            self._knowledge_search_spec(),
            self._ingest_directory_spec(),
            self._get_store_stats_spec(),
        )

    @hookimpl
    async def execute_tool(self, context: Any, tool: ToolSpec, arguments: dict[str, object]) -> ToolResult | None:
        if tool.name == "knowledge_search":
            return await self._knowledge_search(arguments)
        if tool.name == "ingest_directory":
            return await self._ingest_directory(arguments)
        if tool.name == "knowledge_stats":
            return await self._knowledge_stats()
        return None

    # -- Internals --------------------------------------------------------- #
    async def _knowledge_search(self, arguments: dict[str, object]) -> ToolResult:
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            return ToolResult(tool_name="knowledge_search", status=ToolResultStatus.ERROR, error="'query' 参数必填。", output={})

        top_k = int(arguments.get("top_k", 5))
        source_filter = arguments.get("source_filter")
        filter_dict = {"source_file": source_filter} if source_filter and isinstance(source_filter, str) else None

        try:
            query_vector = await self._embed(query)
            results = await self._search_similar(query_vector, top_k=top_k, filter=filter_dict)
        except Exception as exc:
            logger.warning("knowledge_search failed: %s", exc)
            return ToolResult(
                tool_name="knowledge_search",
                status=ToolResultStatus.ERROR,
                error=f"搜索失败: {exc}",
                output={},
            )

        if not results:
            return ToolResult(
                tool_name="knowledge_search",
                status=ToolResultStatus.OK,
                output={"result": "知识库中未找到相关内容。请先使用 ingest_directory 摄入文件。"},
                metadata={"count": 0},
            )

        formatted = "\n\n".join(
            f"[{i+1}] (相似度: {r.score:.3f}) 来源: {r.metadata.get('source_file', 'unknown')}\n{r.content}"
            for i, r in enumerate(results)
        )
        return ToolResult(
            tool_name="knowledge_search",
            status=ToolResultStatus.OK,
            output={"result": formatted},
            metadata={"count": len(results)},
        )

    async def _ingest_directory(self, arguments: dict[str, object]) -> ToolResult:
        path = arguments.get("path")
        if not path or not isinstance(path, str):
            return ToolResult(tool_name="ingest_directory", status=ToolResultStatus.ERROR, error="'path' 参数必填。", output={})

        patterns = arguments.get("patterns", ["*.py", "*.md", "*.txt"])
        if not isinstance(patterns, list):
            patterns = [str(patterns)]
        chunk_size = int(arguments.get("chunk_size", 1000))
        chunk_overlap = int(arguments.get("chunk_overlap", 200))

        files = self._find_files(path, patterns)
        if not files:
            return ToolResult(
                tool_name="ingest_directory",
                status=ToolResultStatus.OK,
                output={"result": f"在 {path} 中未找到匹配的文件。"},
                metadata={"files_ingested": 0, "chunks_created": 0},
            )

        splitter = RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks: list[Chunk] = []

        for file_path in files:
            content = await asyncio.to_thread(read_file_safe, file_path)
            if not content:
                continue
            chunks = splitter.split(content, metadata={"source_file": file_path})
            all_chunks.extend(chunks)

        if not all_chunks:
            return ToolResult(
                tool_name="ingest_directory",
                status=ToolResultStatus.OK,
                output={"result": "未生成任何切片（可能文件均为空或二进制）。"},
                metadata={"files_ingested": 0, "chunks_created": 0},
            )

        # 批量向量化
        texts = [c.content for c in all_chunks]
        try:
            embeddings = await self._embed_texts(texts)
        except Exception as exc:
            logger.warning("ingest_directory embedding failed: %s", exc)
            return ToolResult(
                tool_name="ingest_directory",
                status=ToolResultStatus.ERROR,
                error=f"向量化失败: {exc}",
                output={},
            )

        embedded_chunks = [
            c.model_copy(update={"embedding": emb}) for c, emb in zip(all_chunks, embeddings)
        ]

        # 删除旧数据
        for file_path in files:
            await self._delete_by_source(file_path)

        # 存储
        await self._store_chunks(embedded_chunks)

        stats = await self._get_store_stats()
        return ToolResult(
            tool_name="ingest_directory",
            status=ToolResultStatus.OK,
            output={"result": f"✅ 已摄入 {len(files)} 个文件，生成 {len(all_chunks)} 个切片。知识库总计 {stats.get('total_chunks', 0)} 个切片。"},
            metadata={"files_ingested": len(files), "chunks_created": len(all_chunks)},
        )

    async def _knowledge_stats(self) -> ToolResult:
        stats = await self._get_store_stats()
        return ToolResult(
            tool_name="knowledge_stats",
            status=ToolResultStatus.OK,
            output={"result": f"知识库统计：{stats.get('total_chunks', 0)} 个切片，维度 {stats.get('vector_dim', 0)}。"},
            metadata=stats,
        )

    # -- Hook wrappers ----------------------------------------------------- #
    async def _embed(self, query: str) -> list[float]:
        for value in await _settle(self.harness.bus.pm.hook.embed_query(query=query)):
            if value is not None:
                return value
        return []

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        for value in await _settle(self.harness.bus.pm.hook.embed_texts(texts=texts)):
            if value is not None:
                return value
        return [[] for _ in texts]

    async def _search_similar(self, query_vector: list[float], top_k: int = 5, filter: dict | None = None) -> list[Any]:
        for value in await _settle(self.harness.bus.pm.hook.search_similar(query_vector=query_vector, top_k=top_k, filter=filter)):
            if value is not None:
                return value if isinstance(value, list) else [value]
        return []

    async def _delete_by_source(self, source: str) -> int:
        for value in await _settle(self.harness.bus.pm.hook.delete_by_source(source=source)):
            if value is not None:
                return value
        return 0

    async def _store_chunks(self, chunks: list[Chunk]) -> None:
        await _settle(self.harness.bus.pm.hook.store_chunks(chunks=chunks))

    async def _get_store_stats(self) -> dict[str, Any]:
        for value in await _settle(self.harness.bus.pm.hook.get_store_stats()):
            if value is not None:
                return value
        return {}

    # -- Helpers ----------------------------------------------------------- #
    @staticmethod
    def _find_files(root: str, patterns: list[str]) -> list[str]:
        files: list[str] = []
        for pattern in patterns:
            matched = glob.glob(os.path.join(root, "**", pattern), recursive=True)
            files.extend(m for m in matched if os.path.isfile(m))
        return sorted(set(files))


__all__ = ["KnowledgePlugin"]
