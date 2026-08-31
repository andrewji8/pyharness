"""Tests for streaming tool output (MCP tool streaming batch 1/2).

Covers both the tool-level ``SessionContext.tool_emitter`` contract and the
core-level injection that broadcasts ``EventType.TOOL_STREAM`` on the bus.
"""

from __future__ import annotations

import pytest

from pyharness.context import SessionContext
from pyharness.core import EventBus, Harness
from pyharness.plugins.tool_python_exec import PythonExecPlugin
from pyharness.schema import EventType, HarnessConfig, ToolStreamEvent


def _streaming_code() -> str:
    """Prints two lines with a gap so streaming is observable (no flush=True; relies on PYTHONUNBUFFERED)."""
    return (
        "import time\n"
        "print('Line 1')\n"
        "time.sleep(0.1)\n"
        "print('Line 2')\n"
    )


def _norm(text: str) -> str:
    """Normalize CRLF (Windows subprocess output) to LF for portable asserts."""
    return text.replace("\r\n", "\n")


@pytest.mark.asyncio
async def test_python_exec_streams_via_emitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intermediate stdout lines are pushed through ``tool_emitter``."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "subprocess")

    events: list[ToolStreamEvent] = []

    async def emitter(ev: ToolStreamEvent) -> None:
        events.append(ev)

    ctx = SessionContext(tool_emitter=emitter)
    plugin = PythonExecPlugin()
    spec = plugin.get_tool_specs(ctx)[0]

    result = await plugin.execute_tool(ctx, spec, {"code": _streaming_code()})

    assert result is not None
    assert result.status.value == "ok"
    assert "Line 1" in result.output["stdout"]
    assert "Line 2" in result.output["stdout"]

    contents = [_norm(e.content) for e in events]
    assert "Line 1\n" in contents
    assert "Line 2\n" in contents
    assert contents.index("Line 1\n") < contents.index("Line 2\n")
    assert all(e.stream_type == "stdout" for e in events)
    assert all(e.tool_name == "python_exec" for e in events)


@pytest.mark.asyncio
async def test_python_exec_final_result_keeps_full_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final ToolResult still carries the complete stdout text."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "subprocess")

    async def noop(emitter: "ToolStreamEvent") -> None:  # pragma: no cover
        pass

    ctx = SessionContext(tool_emitter=noop)
    plugin = PythonExecPlugin()
    spec = plugin.get_tool_specs(ctx)[0]

    result = await plugin.execute_tool(ctx, spec, {"code": _streaming_code()})

    assert result is not None
    assert result.status.value == "ok"
    assert "Line 1\nLine 2\n" in _norm(result.output["stdout"])


@pytest.mark.asyncio
async def test_exec_tool_injects_emitter_and_broadcasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core ``_exec_tool`` injects an emitter and emits ``TOOL_STREAM`` on bus."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "subprocess")

    bus = EventBus()
    harness = Harness(event_bus=bus, config=HarnessConfig(auto_load_entry_points=False))
    plugin = PythonExecPlugin()
    harness.register_plugin(plugin)

    events: list[ToolStreamEvent] = []

    def on_stream(event_type: str, context: SessionContext, event: ToolStreamEvent, **_: object) -> None:
        events.append(event)

    bus.on(EventType.TOOL_STREAM.value, on_stream)

    ctx = SessionContext()
    spec = plugin.get_tool_specs(ctx)[0]

    result = await harness._exec_tool(ctx, spec, {"code": _streaming_code()})

    assert result is not None
    assert result.status.value == "ok"
    assert "Line 1\nLine 2\n" in _norm(result.output["stdout"])
    assert any(_norm(e.content) == "Line 1\n" for e in events)
    assert any(_norm(e.content) == "Line 2\n" for e in events)


@pytest.mark.asyncio
async def test_python_exec_emitter_not_buffered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emitter timestamps must be spaced apart, proving no output buffering."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "subprocess")

    timestamps: list[float] = []

    async def emitter(ev: ToolStreamEvent) -> None:
        timestamps.append(ev.timestamp)

    ctx = SessionContext(tool_emitter=emitter)
    plugin = PythonExecPlugin()
    spec = plugin.get_tool_specs(ctx)[0]

    result = await plugin.execute_tool(ctx, spec, {"code": _streaming_code()})

    assert result is not None
    assert result.status.value == "ok"
    assert len(timestamps) >= 2
    gap_ms = (timestamps[1] - timestamps[0]) * 1000
    assert gap_ms >= 50, f"emitter events too close: {gap_ms:.0f}ms (buffered?)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
