"""Tests for Phase 3 parallel subagent orchestration."""

from __future__ import annotations

import pytest
from pluggy import HookimplMarker

from pyharness import Harness
from pyharness.context import SessionContext
from pyharness.core import _settle
from pyharness.plugins.guard_approval import ApprovalGuardPlugin
from pyharness.plugins.llm import entry as llm
from pyharness.plugins.tool_subagent import SubagentToolPlugin
from pyharness.plugins.ui_cli import CLIUIPlugin
from pyharness.schema import (
    AgentConfig,
    HarnessConfig,
    LLMResponse,
    SubagentResult,
    SubagentSpec,
    ToolCall,
    ToolSpec,
)

hookimpl = HookimplMarker("pyharness")


def _harness(*plugins, auto_load: bool = False) -> Harness:
    h = Harness(config=HarnessConfig(auto_load_entry_points=auto_load))
    llm.clear()
    llm.use_dummy(models=("mock-model",), plan=[LLMResponse(model="mock-model", content="Mock LLM response")])
    h.register_plugin(llm)
    for plugin in plugins:
        h.register_plugin(plugin)
    h.initialize()
    return h


async def test_parallel_subagents() -> None:
    """3 subagents run in parallel; verify all complete and order is preserved."""
    import asyncio
    from unittest.mock import patch

    llm.clear()
    llm.use_dummy(models=("mock-model",), plan=[LLMResponse(model="mock-model", content="done")])
    h = _harness(SubagentToolPlugin())
    ctx = SessionContext()

    specs_dict: dict[str, ToolSpec] = {}
    for plugin_specs in h.bus.pm.hook.get_tool_specs(context=ctx):
        specs_dict.update({s.name: s for s in plugin_specs})
    tool = specs_dict["spawn_subagents"]

    arguments = {
        "subagents": [
            {"name": "a", "task": "task a", "model": "mock-model", "max_turns": 2, "timeout": 120.0},
            {"name": "b", "task": "task b", "model": "mock-model", "max_turns": 2, "timeout": 120.0},
            {"name": "c", "task": "task c", "model": "mock-model", "max_turns": 2, "timeout": 120.0},
        ]
    }

    task_group_calls: list[tuple[list, dict]] = []

    original_task_group = asyncio.TaskGroup

    class RecordingTaskGroup(original_task_group):
        def __init__(self, *args, **kwargs):
            task_group_calls.append((args, kwargs))
            super().__init__(*args, **kwargs)

    with patch("asyncio.TaskGroup", RecordingTaskGroup):
        raw = h.bus.pm.hook.execute_tool(context=ctx, tool=tool, arguments=arguments)
        result = next((r for r in await _settle(raw) if r is not None), None)

    assert result is not None
    assert result.status.value == "ok"
    assert result.output["count"] == 3
    names = [r["spec"]["name"] for r in result.output["results"]]
    assert names == ["a", "b", "c"]
    assert len(task_group_calls) >= 1


async def test_subagent_tool_filtering() -> None:
    """When allowed_tools is set, only those tools are inherited."""
    from pyharness.core import Harness

    parent_tools = [
        ToolSpec(name="fs_read", parameters=()),
        ToolSpec(name="fs_write", parameters=()),
        ToolSpec(name="web_fetch", parameters=()),
    ]

    # 直接测试 _filter_tools 方法
    filtered = Harness._filter_tools(parent_tools, allowed=["fs_read"])
    assert len(filtered) == 1
    assert filtered[0].name == "fs_read"
    assert all(t.name != "fs_write" for t in filtered)

    # 测试递归防护：depth 达到 max_depth 时移除 spawn 工具
    tools_with_spawn = parent_tools + [ToolSpec(name="spawn_subagents", parameters=())]
    filtered_deep = Harness._filter_tools(tools_with_spawn, allowed=None, depth=3, max_depth=3)
    assert all(t.name != "spawn_subagents" for t in filtered_deep)


async def test_subagent_error_isolation() -> None:
    """One crashing subagent must not affect the others."""
    from unittest.mock import patch

    llm.clear()
    llm.use_dummy(models=("mock-model",), plan=[LLMResponse(model="mock-model", content="ok")])

    h = _harness(SubagentToolPlugin())
    ctx = SessionContext()
    specs_dict: dict[str, ToolSpec] = {}
    for plugin_specs in h.bus.pm.hook.get_tool_specs(context=ctx):
        specs_dict.update({s.name: s for s in plugin_specs})
    tool = specs_dict["spawn_subagents"]

    arguments = {
        "subagents": [
            {"name": "good", "task": "ok task", "model": "mock-model", "max_turns": 2, "timeout": 120.0},
            {"name": "bad", "task": "bad task", "model": "mock-model", "max_turns": 2, "timeout": 120.0},
        ]
    }

    original_spawn = h.spawn_subagent

    async def mock_spawn(spec, parent_tools=None, parent_config=None):
        if spec.name == "bad":
            raise RuntimeError("模拟崩溃")
        return await original_spawn(spec, parent_tools=parent_tools, parent_config=parent_config)

    with patch.object(h, "spawn_subagent", mock_spawn):
        raw = h.bus.pm.hook.execute_tool(context=ctx, tool=tool, arguments=arguments)
        result = next((r for r in await _settle(raw) if r is not None), None)

    assert result is not None
    assert result.status.value == "ok"
    results = result.output["results"]
    assert len(results) == 2
    assert results[0]["status"] == "ok"
    assert results[1]["status"] == "error"
    assert "模拟崩溃" in results[1]["error"]


async def test_subagent_guard_still_active() -> None:
    """Subagent executing a guarded tool still triggers pre_tool_execution."""
    from unittest.mock import patch

    guard_called = False

    async def mock_ask_user(prompt: str, metadata: dict) -> bool:
        nonlocal guard_called
        guard_called = True
        return False

    class MockPythonExecPlugin:
        @hookimpl
        def get_tool_specs(self, context: SessionContext) -> tuple[ToolSpec, ...]:
            return (ToolSpec(name="python_exec", parameters=()),)

    guard = ApprovalGuardPlugin(default_action="reject")
    ui = CLIUIPlugin(timeout=0.1)
    python_plugin = MockPythonExecPlugin()

    h = _harness(guard, ui, SubagentToolPlugin(), python_plugin)

    llm.clear()
    llm.use_dummy(
        models=("mock-model",),
        plan=[
            LLMResponse(
                model="mock-model",
                content="",
                tool_calls=(ToolCall(id="c1", tool_name="python_exec", arguments={"code": "1+1"}),),
            ),
            LLMResponse(model="mock-model", content="done"),
        ],
    )

    spec = SubagentSpec(
        name="guarded-worker",
        task="execute python code",
        model="mock-model",
        allowed_tools=["python_exec"],
        max_turns=3,
        timeout=120.0,
    )

    with patch.object(guard, "_ask_user", mock_ask_user):
        result = await h.spawn_subagent(
            spec,
            parent_tools=[ToolSpec(name="python_exec", parameters=())],
            parent_config=AgentConfig(name="p", model="mock-model"),
        )

    assert guard_called is True
    assert result.status == "ok"
