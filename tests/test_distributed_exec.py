"""Tests for distributed tool execution (Redis task queue + worker).

Uses fakeredis so no real Redis is required. Covers: enqueue→worker→result,
timeout when no worker consumes, worker-exception → error result, and graceful
degradation to in-process execution when the switch is off.
"""

from __future__ import annotations

import asyncio

import pytest

for _dep in ("redis", "fakeredis.aioredis"):
    try:
        __import__(_dep)
    except ImportError:
        pytest.skip(f"{_dep} not installed; skipping distributed exec tests", allow_module_level=True)

from pluggy import HookimplMarker

from pyharness import Harness
from pyharness.distributed import (
    TASK_QUEUE_KEY,
    ToolTask,
    ToolTaskResult,
    await_tool_result,
    deserialize_result,
    deserialize_task,
    distributed_exec_enabled,
    enqueue_tool_task,
    process_one,
    serialize_result,
    serialize_task,
)
from pyharness.schema import (
    HarnessConfig,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)

_hook = HookimplMarker("pyharness")

ECHO_SPEC = ToolSpec(name="echo", description="echo input", timeout_seconds=5.0)


class _EchoExecutor:
    """A fake tool executor that returns a fixed result (or raises)."""

    def __init__(self, result: ToolResult | None = None, exc: Exception | None = None) -> None:
        self.result = result or ToolResult(tool_name="echo", output={"echoed": "hi"})
        self.exc = exc
        self.calls = 0

    @_hook
    def get_tool_specs(self, context) -> tuple[ToolSpec, ...]:
        return (ECHO_SPEC,)

    @_hook
    async def execute_tool(self, context, tool, arguments) -> ToolResult | None:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if tool.name != "echo":
            return None
        return self.result


def _worker_harness(executor=None) -> Harness:
    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    h.register_plugin(executor or _EchoExecutor())
    return h


def _make_task(**kwargs) -> ToolTask:
    defaults = dict(tool=ECHO_SPEC, arguments={"text": "hi"}, timeout=2.0)
    defaults.update(kwargs)
    return ToolTask(**defaults)


@pytest.fixture()
def shared_client():
    import fakeredis
    import fakeredis.aioredis

    server = fakeredis.FakeServer()
    return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)


async def test_task_serialization_round_trip() -> None:
    """ToolTask / ToolTaskResult serialize and deserialize losslessly."""
    task = _make_task(arguments={"nested": {"a": 1}, "flag": True})
    raw = serialize_task(task)
    decoded = deserialize_task(raw)
    assert decoded.task_id == task.task_id
    assert decoded.tool.name == "echo"
    assert decoded.arguments == {"nested": {"a": 1}, "flag": True}
    assert decoded.timeout == 2.0

    result = ToolResult(tool_name="echo", output={"k": "v"}, status=ToolResultStatus.OK)
    res_raw = serialize_result(task.task_id, result)
    res_dec = deserialize_result(res_raw)
    assert isinstance(res_dec, ToolTaskResult)
    assert res_dec.task_id == task.task_id
    assert res_dec.result.output == {"k": "v"}


async def test_enqueue_worker_execute_web_receives_result(shared_client) -> None:
    """Full loop: web enqueues → worker executes → web gets the ToolResult."""
    task = _make_task()
    await enqueue_tool_task(shared_client, task)

    wh = _worker_harness(_EchoExecutor(result=ToolResult(tool_name="echo", output={"echoed": "hello"})))
    consumed = await process_one(shared_client, wh, poll_timeout=0.2)
    assert consumed is True

    result = await await_tool_result(shared_client, task.task_id, timeout=0.5)
    assert result.status == ToolResultStatus.OK
    assert result.output == {"echoed": "hello"}
    assert result.tool_name == "echo"
    assert wh._plugin_registry  # sanity: harness usable


async def test_timeout_when_no_worker_consumes(shared_client) -> None:
    """No worker present → web node times out and gets an error ToolResult."""
    task = _make_task(timeout=0.3)
    await enqueue_tool_task(shared_client, task)
    result = await asyncio.wait_for(await_tool_result(shared_client, task.task_id, timeout=0.3), timeout=2.0)
    assert result.status == ToolResultStatus.ERROR
    assert "timed out" in (result.error or "")


async def test_worker_exception_yields_error_result(shared_client) -> None:
    """Worker raising mid-execution must still publish an error result."""
    task = _make_task()
    await enqueue_tool_task(shared_client, task)
    exc = RuntimeError("boom")
    wh = _worker_harness(_EchoExecutor(exc=exc))
    consumed = await process_one(shared_client, wh, poll_timeout=0.2)
    assert consumed is True

    result = await await_tool_result(shared_client, task.task_id, timeout=0.5)
    assert result.status == ToolResultStatus.ERROR
    assert "boom" in (result.error or "")


def test_distributed_enabled_requires_switch_and_url(monkeypatch) -> None:
    """distributed_exec_enabled() gates on PYHARNESS_DISTRIBUTED_EXEC + REDIS_URL."""
    monkeypatch.delenv("PYHARNESS_DISTRIBUTED_EXEC", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert distributed_exec_enabled() is False

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    assert distributed_exec_enabled() is False  # switch off

    monkeypatch.setenv("PYHARNESS_DISTRIBUTED_EXEC", "1")
    assert distributed_exec_enabled() is True

    monkeypatch.delenv("REDIS_URL")
    assert distributed_exec_enabled() is False  # url missing


async def test_degradation_in_process_exec_unchanged(monkeypatch) -> None:
    """With the switch off, _exec_tool runs in-process (no Redis involved)."""
    monkeypatch.delenv("PYHARNESS_DISTRIBUTED_EXEC", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    executor = _EchoExecutor(result=ToolResult(tool_name="echo", output={"ok": True}))
    h.register_plugin(executor)
    h.initialize()

    from pyharness.context import SessionContext
    from pyharness.core import _distributed_enabled

    assert _distributed_enabled() is False
    ctx = SessionContext()
    result = await h._exec_tool(ctx, ECHO_SPEC, {"text": "hi"})
    # In-process: the local executor ran and returned its result locally.
    assert executor.calls == 1
    assert result.status == ToolResultStatus.OK
    assert result.output == {"ok": True}


def _shared_pair():
    """Two fakeredis clients sharing one FakeServer (web + worker)."""
    import fakeredis
    import fakeredis.aioredis

    server = fakeredis.FakeServer()
    web = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    wkr = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return web, wkr


async def test_core_exec_tool_offloads_when_enabled(monkeypatch) -> None:
    """Integration: Harness._exec_tool offloads to a worker when enabled."""
    monkeypatch.setenv("PYHARNESS_DISTRIBUTED_EXEC", "1")
    monkeypatch.setenv("REDIS_URL", "redis://fake")

    web_client, worker_client = _shared_pair()
    import redis.asyncio as real_redis_asyncio

    # _exec_tool_distributed creates its client via redis.asyncio.from_url.
    monkeypatch.setattr(real_redis_asyncio, "from_url", lambda *a, **k: web_client)

    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    executor = _EchoExecutor()
    h.register_plugin(executor)
    h.initialize()

    worker_executor = _EchoExecutor(result=ToolResult(tool_name="echo", output={"echoed": "via-worker"}))
    wh = _worker_harness(worker_executor)

    from pyharness.context import SessionContext

    ctx = SessionContext()
    exec_task = asyncio.create_task(h._exec_tool(ctx, ECHO_SPEC, {"text": "hi"}))
    try:
        # Wait until the web node has actually enqueued its task, then let the
        # worker consume + process exactly one. Deterministic (no infinite task).
        for _ in range(200):
            if await worker_client.llen(TASK_QUEUE_KEY) > 0:
                break
            await asyncio.sleep(0.01)
        consumed = await process_one(worker_client, wh, poll_timeout=0.2)
        assert consumed is True
        result = await asyncio.wait_for(exec_task, timeout=5.0)
    finally:
        exec_task.cancel()
        await asyncio.gather(exec_task, return_exceptions=True)

    assert result.status == ToolResultStatus.OK
    assert result.output == {"echoed": "via-worker"}
    # The web node's OWN executor must NOT have run — work happened on the worker.
    assert executor.calls == 0
    assert worker_executor.calls == 1
