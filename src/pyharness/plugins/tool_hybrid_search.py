"""Hybrid search tool for PyHarness RAG.

Combines FTS5 keyword search with vector semantic search using
Reciprocal Rank Fusion (RRF) to produce a unified, ranked result set.
"""

from __future__ import annotations

import logging
from typing import Any

from pluggy import HookimplMarker

from pyharness.core import _settle
from pyharness.schema import HybridSearchInput, HybridSearchResult, ToolArg, ToolResult, ToolResultStatus, ToolSpec
from pyharness.specs import AgentHooks

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")


class HybridSearchPlugin:
    """Tool provider for hybrid (FTS5 + vector) search."""

    def __init__(self) -> None:
        self.harness: Any = None

    @hookimpl
    def harness_initialized(self, harness: Any) -> None:
        self.harness = harness

    def _hybrid_search_spec(self) -> ToolSpec:
        return ToolSpec(
            name="hybrid_search",
            description=(
                "混合搜索：结合 FTS5 关键词匹配和向量语义检索。"
                "FTS5 擅长精确关键词匹配，向量擅长语义理解。"
                "通过 RRF 加权融合得到最终排序。"
                "适用于不确定该用关键词还是语义搜索时的通用选择。"
            ),
            parameters=(
                ToolArg(name="query", type="string", description="搜索查询。", required=True),
                ToolArg(name="top_k", type="integer", description="返回结果数量（默认 5）。", required=False),
                ToolArg(
                    name="fts_weight",
                    type="number",
                    description="FTS5 权重（0~1，默认 0.3）。",
                    required=False,
                ),
                ToolArg(
                    name="vector_weight",
                    type="number",
                    description="向量权重（0~1，默认 0.7）。",
                    required=False,
                ),
            ),
            timeout_seconds=30.0,
        )

    @hookimpl
    def get_tool_specs(self, context: Any) -> tuple[ToolSpec, ...]:
        return (self._hybrid_search_spec(),)

    @hookimpl
    async def execute_tool(self, context: Any, tool: ToolSpec, arguments: dict[str, object]) -> ToolResult | None:
        if tool.name != "hybrid_search":
            return None
        return await self._hybrid_search(arguments)

    async def _hybrid_search(self, arguments: dict[str, object]) -> ToolResult:
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            return ToolResult(tool_name="hybrid_search", status=ToolResultStatus.ERROR, error="'query' 参数必填。", output={})

        top_k = int(arguments.get("top_k", 5))
        fts_weight = float(arguments.get("fts_weight", 0.3))
        vector_weight = float(arguments.get("vector_weight", 0.7))

        # 1. FTS5 检索
        fts_results: list[Any] = []
        try:
            for value in await _settle(
                self.harness.bus.pm.hook.search_session(session_id="*", query=query, limit=top_k * 2)
            ):
                if value is not None:
                    fts_results.extend(value if isinstance(value, list) else [value])
        except Exception as exc:
            logger.debug("hybrid_search FTS5 failed: %s", exc)

        # 2. 向量检索
        vector_results: list[Any] = []
        try:
            query_vector: list[float] = []
            for value in await _settle(self.harness.bus.pm.hook.embed_query(query=query)):
                if value is not None:
                    query_vector = value
                    break
            for value in await _settle(
                self.harness.bus.pm.hook.search_similar(query_vector=query_vector, top_k=top_k * 2)
            ):
                if value is not None:
                    vector_results.extend(value if isinstance(value, list) else [value])
        except Exception as exc:
            logger.warning("hybrid_search vector failed: %s", exc)

        # 3. RRF 融合
        k = 60
        scores: dict[str, dict[str, Any]] = {}

        for rank, r in enumerate(fts_results):
            cid = getattr(r, "chunk_id", getattr(r, "session_id", str(rank)))
            content = getattr(r, "content", getattr(r, "snippet", ""))
            meta = getattr(r, "metadata", {}) or {}
            if cid not in scores:
                scores[cid] = {"content": content, "metadata": meta, "score": 0.0}
            scores[cid]["score"] += fts_weight / (k + rank)
            scores[cid]["fts_score"] = scores[cid].get("fts_score", 0.0) + fts_weight / (k + rank)

        for rank, r in enumerate(vector_results):
            cid = getattr(r, "chunk_id", str(rank))
            content = getattr(r, "content", "")
            meta = getattr(r, "metadata", {}) or {}
            if cid not in scores:
                scores[cid] = {"content": content, "metadata": meta, "score": 0.0}
            scores[cid]["score"] += vector_weight / (k + rank)
            scores[cid]["vector_score"] = scores[cid].get("vector_score", 0.0) + vector_weight / (k + rank)
            scores[cid]["content"] = content
            scores[cid]["metadata"] = meta

        sorted_items = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
        results = [
            HybridSearchResult(
                chunk_id=cid,
                content=data.get("content", ""),
                fts_score=data.get("fts_score", 0.0),
                vector_score=data.get("vector_score", 0.0),
                hybrid_score=data.get("score", 0.0),
                metadata=data.get("metadata", {}),
            )
            for cid, data in sorted_items
        ]

        if not results:
            return ToolResult(
                tool_name="hybrid_search",
                status=ToolResultStatus.OK,
                output={"result": "混合搜索未找到相关内容。"},
                metadata={"count": 0},
            )

        formatted = "\n\n".join(
            f"[{i+1}] (混合分数: {r.hybrid_score:.3f}, FTS: {r.fts_score:.3f}, 向量: {r.vector_score:.3f}) "
            f"来源: {r.metadata.get('source_file', 'unknown')}\n{r.content}"
            for i, r in enumerate(results)
        )
        return ToolResult(
            tool_name="hybrid_search",
            status=ToolResultStatus.OK,
            output={"result": formatted},
            metadata={"count": len(results)},
        )


__all__ = ["HybridSearchPlugin"]
