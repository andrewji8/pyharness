"""Subagent tool plugin for PyHarness.

Provides the ``spawn_subagents`` tool, allowing the LLM to spawn multiple
worker subagents in parallel and collect their results.

Design
------
* The plugin captures the harness reference via ``harness_initialized`` so
  it can delegate to ``Harness.spawn_subagent``.
* Parallelism is implemented with ``asyncio.TaskGroup`` (Python 3.11+).
* Each subagent receives the parent context so it can reference prior history.
* Results are returned as a single structured tool result, ready to be
  injected back into the parent conversation.
* [Fix] Subagents use the real OpenRouter model (never the dummy provider).
* [Fix] Each subagent runs in an isolated open_session scope (no deadlocks).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from pluggy import HookimplMarker

from pyharness.context import SessionContext, open_session
from pyharness.schema import AgentConfig, SubagentResult, SubagentSpec, ToolArg, ToolResult, ToolResultStatus, ToolSpec

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")

# 主会话使用的真实模型（OpenRouter）。子 Agent 严禁回退到 dummy provider。
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


class SubagentToolPlugin:
    """Tool provider for parallel subagent orchestration."""

    def __init__(self) -> None:
        self.harness: Any = None

    @hookimpl
    def harness_initialized(self, harness: Any) -> None:
        """Capture the harness reference after entry-points are loaded."""
        self.harness = harness

    def _spawn_subagents_spec(self) -> ToolSpec:
        return ToolSpec(
            name="spawn_subagents",
            description=(
                "并行派生多个子 Agent 执行独立子任务，等待全部完成后汇总结果。"
                "适用于需要同时调研多个主题、并行处理多个文件的场景。"
                "参数 subagents 是一个数组，每个元素包含 name, task, model, max_turns, timeout。"
            ),
            parameters=(
                ToolArg(
                    name="subagents",
                    type="array",
                    description="要并行派生的子 Agent 规格数组。每个元素需包含 name, task, model, max_turns, timeout。",
                    required=True,
                ),
            ),
            timeout_seconds=300.0,
        )

    @hookimpl
    def get_tool_specs(self, context: SessionContext) -> tuple[ToolSpec, ...]:
        return (self._spawn_subagents_spec(),)

    @hookimpl
    async def execute_tool(
        self, context: SessionContext, tool: ToolSpec, arguments: dict[str, object]
    ) -> ToolResult | None:
        if tool.name != "spawn_subagents":
            return None

        raw_specs = arguments.get("subagents", [])
        if not isinstance(raw_specs, list):
            return ToolResult(
                tool_name="spawn_subagents",
                status=ToolResultStatus.ERROR,
                error="'subagents' 参数必须是数组。",
                output={},
            )
        if not raw_specs:
            return ToolResult(
                tool_name="spawn_subagents",
                status=ToolResultStatus.OK,
                output={"count": 0, "results": []},
            )

        try:
            specs = [SubagentSpec.model_validate(s) for s in raw_specs]
        except Exception as exc:
            return ToolResult(
                tool_name="spawn_subagents",
                status=ToolResultStatus.ERROR,
                error=f"子 Agent 规格校验失败: {exc}",
                output={"raw": raw_specs},
            )

        if self.harness is None:
            return ToolResult(
                tool_name="spawn_subagents",
                status=ToolResultStatus.ERROR,
                error="Harness 未初始化，无法派生子 Agent。",
                output={},
            )

        try:
            results = await self._spawn_parallel(specs, context)
        except Exception as exc:
            logger.exception("spawn_subagents failed")
            return ToolResult(
                tool_name="spawn_subagents",
                status=ToolResultStatus.ERROR,
                error=f"并行派生子 Agent 失败: {exc}",
                output={},
            )

        return ToolResult(
            tool_name="spawn_subagents",
            status=ToolResultStatus.OK,
            output={
                "count": len(results),
                "results": [r.model_dump(mode="json") for r in results],
            },
        )

    async def _spawn_parallel(
        self,
        specs: list[SubagentSpec],
        parent_ctx: SessionContext,
    ) -> list[SubagentResult]:
        """Spawn multiple subagents in parallel using ``asyncio.TaskGroup``.

        Each subagent is wrapped in ``_safe_spawn`` so that one failure does
        not crash the whole batch. Results preserve the input order.
        """
        tools: list[ToolSpec] = []
        for plugin_specs in self.harness.bus.pm.hook.get_tool_specs(context=parent_ctx):
            tools.extend(plugin_specs)

        # 【修复 1】排除 spawn_subagents 自身，防止子 Agent 无限递归派生
        tools = [t for t in tools if t.name != "spawn_subagents"]
        logger.info("[subagent] parent tools collected: %s", [t.name for t in tools])

        # 【修复 2】子 Agent 必须用真实模型，严禁回退到 dummy provider
        model = (
            getattr(parent_ctx, "model", None)
            or os.environ.get("OPENROUTER_MODEL")
            or os.environ.get("LLM_MODEL")
            or DEFAULT_MODEL
        )
        logger.info("[subagent] using model=%s", model)

        parent_config = AgentConfig(
            name=getattr(parent_ctx, "agent_name", "parent"),
            model=model,
        )

        async def _run_one(index: int, spec: SubagentSpec) -> SubagentResult:
            try:
                return await asyncio.wait_for(
                    self._safe_spawn(spec, tools, parent_config),
                    timeout=spec.timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("[subagent] %s timed out", spec.name)
                return SubagentResult(
                    spec=spec,
                    status="timeout",
                    output=None,
                    error=f"超时（{spec.timeout}s）",
                    duration_seconds=spec.timeout,
                    session_id="",
                )
            except Exception as exc:
                # 【修复 3】捕获所有异常，防止 TaskGroup 崩溃导致界面卡死
                logger.exception("[subagent] %s crashed", spec.name)
                return SubagentResult(
                    spec=spec,
                    status="error",
                    output=None,
                    error=str(exc),
                    duration_seconds=0.0,
                    session_id="",
                )

        results: list[SubagentResult] = []
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(_run_one(index, spec))
                for index, spec in enumerate(specs)
            ]
        for task in tasks:
            results.append(task.result())
        return results

    async def _safe_spawn(
        self,
        spec: SubagentSpec,
        parent_tools: list[ToolSpec],
        parent_config: AgentConfig,
    ) -> SubagentResult:
        """Wrap ``spawn_subagent`` with error isolation and ContextVar isolation."""
        try:
            # 【修复 4】open_session 隔离 ContextVar，切断与外层 chat 的关联，防嵌套死锁
            async with open_session(namespace=f"subagent:{spec.name}"):
                return await self.harness.spawn_subagent(
                    spec, parent_tools=parent_tools, parent_config=parent_config
                )
        except Exception as exc:
            logger.exception("Subagent %s failed", spec.name)
            return SubagentResult(
                spec=spec,
                status="error",
                error=str(exc),
                session_id="",
                duration_seconds=0.0,
            )


__all__ = ["SubagentToolPlugin"]