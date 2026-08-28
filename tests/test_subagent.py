"""Tests for the Phase 3 subagent (multi-agent) system."""

from __future__ import annotations

import asyncio
import time

import pytest

from pluggy import HookimplMarker

hookimpl = HookimplMarker("pyharness")

from pyharness import Harness
from pyharness.context import SessionContext, _subagent_depth
from pyharness.core import _settle
from pyharness.plugins.llm import entry as llm
from pyharness.plugins.tool_subagent import SubagentToolPlugin
from pyharness.schema import (
    AgentConfig,
    HarnessConfig,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    SubagentResult,
    SubagentSpec,
    ToolSpec,
)


def _harness(*plugins, auto_load: bool = False) -> Harness:
    h = Harness(config=HarnessConfig(auto_load_entry_points=auto_load))
    llm.clear()
    llm.use_dummy(models=("dummy",), plan=[LLMResponse(model="dummy", content="worker done")])
    h.register_plugin(llm)
    for plugin in plugins:
        h.register_plugin(plugin)
    h.initialize()
    return h


async def test_spawn_subagent_returns_result() -> None:
    h = _harness()
    parent = AgentConfig(name="parent", model="dummy")
    spec = SubagentSpec(name="worker", task="do something", model="dummy", max_turns=2, timeout=120.0)

    result = await h.spawn_subagent(spec)
    assert isinstance(result, SubagentResult)
    assert result.status == "ok"
    assert result.output == "worker done"
    assert result.spec.name == "worker"


async def test_subagent_depth_limit_prevents_recursion() -> None:
    h = _harness()
    parent = AgentConfig(name="parent", model="dummy")
    spec = SubagentSpec(name="deep-worker", task="spawn another", model="dummy", max_turns=1, timeout=120.0)

    token = _subagent_depth.set(3)
    try:
        result = await h.spawn_subagent(spec)
        assert result.status == "error"
        assert result.output is None
        assert "depth limit" in result.error.lower()
    finally:
        _subagent_depth.reset(token)


async def test_subagent_timeout() -> None:
    from pyharness.plugins.llm.dummy import DummyProvider

    slow_plan = [LLMResponse(model="dummy", content="")]
    llm.clear()
    llm.register_provider(DummyProvider(models=("dummy",), plan=slow_plan, delay=15.0))
    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    h.register_plugin(llm)
    h.initialize()

    spec = SubagentSpec(name="slow-worker", task="be slow", model="dummy", max_turns=10, timeout=10.0)
    result = await h.spawn_subagent(spec)
    assert result.status == "timeout"
    assert result.output is None
    assert "timed out" in result.error.lower()


async def test_subagent_finished_hook_fires() -> None:
    finished: list[SubagentResult] = []

    class Collector:
        @hookimpl
        async def subagent_finished(self, result: SubagentResult) -> None:
            finished.append(result)

    h = _harness(Collector())
    spec = SubagentSpec(name="w", task="t", model="dummy", max_turns=1, timeout=120.0)
    await h.spawn_subagent(spec)
    assert len(finished) == 1
    assert finished[0].spec.name == "w"
    assert finished[0].status == "ok"
    assert finished[0].output == "worker done"


async def test_spawn_subagent_uses_hook_when_available() -> None:
    """If a plugin implements spawn_subagent, the engine should use it."""

    class CustomSpawner:
        def __init__(self) -> None:
            self.called = False

        @hookimpl
        def spawn_subagent(self, spec, parent_tools, parent_config):
            self.called = True
            return SubagentResult(spec=spec, status="ok", output="custom", session_id="custom-1")

    h = _harness()
    sp = CustomSpawner()
    h.register_plugin(sp)
    spec = SubagentSpec(name="custom", task="t", model="dummy")
    tools: list[ToolSpec] = []
    config = AgentConfig(name="parent", model="dummy")
    result = await h.spawn_subagent(spec, parent_tools=tools, parent_config=config)
    assert sp.called is True
    assert result.output == "custom"


async def test_spawn_subagents_tool_is_registered() -> None:
    h = _harness(SubagentToolPlugin())
    specs: dict[str, ToolSpec] = {}
    for plugin_specs in h.bus.pm.hook.get_tool_specs(context=SessionContext()):
        specs.update({s.name: s for s in plugin_specs})
    assert "spawn_subagents" in specs


async def test_spawn_subagents_parallel_execution() -> None:
    llm.clear()
    llm.use_dummy(models=("dummy",), plan=[LLMResponse(model="dummy", content="worker done")])
    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    h.register_plugin(llm)
    h.register_plugin(SubagentToolPlugin())
    h.initialize()

    ctx = SessionContext()
    specs: dict[str, ToolSpec] = {}
    for plugin_specs in h.bus.pm.hook.get_tool_specs(context=ctx):
        specs.update({s.name: s for s in plugin_specs})
    tool = specs["spawn_subagents"]

    arguments = {
        "subagents": [
            {"name": "w1", "task": "task 1", "model": "dummy", "max_turns": 2, "timeout": 120.0},
            {"name": "w2", "task": "task 2", "model": "dummy", "max_turns": 2, "timeout": 120.0},
        ]
    }

    raw = h.bus.pm.hook.execute_tool(context=ctx, tool=tool, arguments=arguments)
    result = next((r for r in await _settle(raw) if r is not None), None)
    assert result is not None
    assert result.status.value == "ok"
    assert result.output["count"] == 2
    assert len(result.output["results"]) == 2
    outputs = [r["output"] for r in result.output["results"]]
    assert "worker done" in outputs


async def test_spawn_subagents_validates_input() -> None:
    llm.clear()
    llm.use_dummy(models=("dummy",), plan=[LLMResponse(model="dummy", content="worker done")])
    h = _harness(SubagentToolPlugin())
    ctx = SessionContext()
    specs: dict[str, ToolSpec] = {}
    for plugin_specs in h.bus.pm.hook.get_tool_specs(context=ctx):
        specs.update({s.name: s for s in plugin_specs})
    tool = specs["spawn_subagents"]

    raw = h.bus.pm.hook.execute_tool(context=ctx, tool=tool, arguments={"subagents": "not-a-list"})
    result = next((r for r in await _settle(raw) if r is not None), None)
    assert result is not None
    assert result.status.value == "error"


async def test_subagent_inherits_parent_tools() -> None:
    """Subagent LLM request should include parent_tools (not an empty list)."""
    llm.clear()
    llm.use_dummy(models=("dummy",), plan=[LLMResponse(model="dummy", content="done")])
    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    h.register_plugin(llm)
    h.initialize()

    parent_tool = ToolSpec(name="python_exec", description="Run Python", parameters=())
    captured_requests: list[LLMRequest] = []

    class CaptureLLM:
        @hookimpl
        async def llm_complete(self, context: SessionContext, request: LLMRequest):
            captured_requests.append(request)
            return LLMResponse(model=request.model, content="done")

    h.register_plugin(CaptureLLM())

    spec = SubagentSpec(name="worker", task="do math", model="dummy", max_turns=2, timeout=120.0)
    result = await h.spawn_subagent(
        spec,
        parent_tools=[parent_tool],
        parent_config=AgentConfig(name="parent", model="dummy"),
    )
    assert result.status == "ok"
    assert len(captured_requests) >= 1
    tool_names = [t["function"]["name"] for r in captured_requests for t in (r.tools or [])]
    assert "python_exec" in tool_names


async def test_subagent_empty_allowed_tools_inherits_all() -> None:
    """When allowed_tools is an empty list, subagent should still inherit all parent_tools."""
    llm.clear()
    llm.use_dummy(models=("dummy",), plan=[LLMResponse(model="dummy", content="done")])
    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    h.register_plugin(llm)
    h.initialize()

    parent_tool = ToolSpec(name="python_exec", description="Run Python", parameters=())
    captured_requests: list[LLMRequest] = []

    class CaptureLLM:
        @hookimpl
        async def llm_complete(self, context: SessionContext, request: LLMRequest):
            captured_requests.append(request)
            return LLMResponse(model=request.model, content="done")

    h.register_plugin(CaptureLLM())

    spec = SubagentSpec(name="worker", task="do math", model="dummy", max_turns=2, timeout=120.0, allowed_tools=[])
    result = await h.spawn_subagent(
        spec,
        parent_tools=[parent_tool],
        parent_config=AgentConfig(name="parent", model="dummy"),
    )
    assert result.status == "ok"
    assert len(captured_requests) >= 1
    tool_names = [t["function"]["name"] for r in captured_requests for t in (r.tools or [])]
    assert "python_exec" in tool_names


if __name__ == "__main__":
    asyncio.run(test_spawn_subagent_returns_result())
    print("PASS: test_spawn_subagent_returns_result")
    asyncio.run(test_subagent_depth_limit_prevents_recursion())
    print("PASS: test_subagent_depth_limit_prevents_recursion")
    asyncio.run(test_subagent_timeout())
    print("PASS: test_subagent_timeout")
    asyncio.run(test_subagent_finished_hook_fires())
    print("PASS: test_subagent_finished_hook_fires")
    asyncio.run(test_spawn_subagent_uses_hook_when_available())
    print("PASS: test_spawn_subagent_uses_hook_when_available")
