"""End-to-end tests for the PyHarness thin core.

Covers the four Design Principles the engine must not regress on:

1. Thin Core / Everything-is-a-Plugin  — capabilities arrive via ``@hookimpl``.
2. Schema-Driven                       — every payload is a frozen pydantic model.
3. Reversibility                       — state transitions are snapshot copies.
4. Async First + contextvars isolation — concurrency-safe per-task contexts.
"""

from __future__ import annotations

import asyncio

import pytest
from pluggy import HookimplMarker

from pyharness import EventBus, Harness
from pyharness.context import (
    SessionContext,
    current_context,
    open_session,
    require_context,
)
from pyharness.schema import (
    AgentConfig,
    Event,
    EventType,
    HarnessConfig,
    LLMResponse,
    Message,
    Role,
    ToolArg,
    ToolResult,
    ToolSpec,
)

hookimpl = HookimplMarker("pyharness")

ECHO_SPEC = ToolSpec(
    name="echo",
    description="Echo text.",
    parameters=(ToolArg(name="text", type="string", required=True),),
)


# --------------------------------------------------------------------------- #
# A fake third-party "plugin" — the only way functionality reaches the engine.
# --------------------------------------------------------------------------- #
class DreamPlugin:
    """Stands in for an LLM provider plugin that also executes the echo tool."""

    def __init__(self, steps: list[LLMResponse]) -> None:
        self.steps = list(steps)
        self.events: list[Event] = []
        self.started = 0
        self.finished = 0

    @hookimpl
    def get_tool_specs(self, context) -> tuple[ToolSpec, ...]:
        return (ECHO_SPEC,)

    @hookimpl
    async def execute_tool(self, context, tool, arguments) -> ToolResult | None:
        if tool.name == "echo":
            return ToolResult(tool_name=tool.name, output={"echo": arguments.get("text", "")})
        return None

    @hookimpl
    async def agent_next(self, context, agent) -> LLMResponse | None:
        return self.steps.pop(0) if self.steps else None

    @hookimpl
    def observe(self, context, event) -> None:
        self.events.append(event)

    @hookimpl
    def session_started(self, context, agent=None) -> None:
        self.started += 1

    @hookimpl
    def session_finished(self, context) -> None:
        self.finished += 1


# --------------------------------------------------------------------------- #
# Reversibility (principle 3)
# --------------------------------------------------------------------------- #
def test_session_context_is_immutable_and_reversible() -> None:
    ctx = SessionContext(namespace="t")
    child = ctx.append_message(Message(role=Role.USER, content="hi"))

    # original snapshot untouched
    assert ctx.messages == ()
    # child gained a message
    assert child.last_message is not None and child.last_message.content == "hi"
    # branching lineage recorded
    assert child.parent_id == ctx.branch_id
    assert child.branch_id != ctx.branch_id

    # memory fold is also reversible
    grown = child.with_memory(cwd="/tmp")
    assert grown.memory == {"cwd": "/tmp"}
    assert child.memory == {}
    assert grown.parent_id == child.branch_id

    # non-branching derive keeps the same branch identity (linear extension)
    same_branch = child.derive(branch=False, memory={"cwd": "/tmp"})
    assert same_branch.branch_id == child.branch_id  # no fork created
    assert same_branch.parent_id == child.parent_id  # branch lineage preserved


# --------------------------------------------------------------------------- #
# contextvars isolation (principle: Async First)
# --------------------------------------------------------------------------- #
async def test_concurrent_sessions_are_isolated() -> None:
    async def worker(tag: str) -> str:
        async with open_session(memory={"tag": tag}) as ctx:
            # require_context sees the per-task binding, not a global
            assert require_context().session_id == ctx.session_id
            await asyncio.sleep(0.01)  # force interleaving
            return f"{tag}:{require_context().session_id}"

    a, b = await asyncio.gather(worker("alpha"), worker("beta"))
    assert a.startswith("alpha:") and b.startswith("beta:")
    assert a.split(":")[1] != b.split(":")[1]


async def test_context_is_reset_after_open_session() -> None:
    assert current_context() is None
    async with open_session() as _ctx:
        assert current_context() is not None
    assert current_context() is None


# --------------------------------------------------------------------------- #
# pluggy event bus + firstresult + run_session (principles 1 & 2)
# --------------------------------------------------------------------------- #
async def test_run_session_drives_loop_with_tool_calls() -> None:
    harness = Harness(config=HarnessConfig(auto_load_entry_points=False))
    plugin = DreamPlugin(
        steps=[
            LLMResponse(
                model="dream",
                content="",
                tool_calls=[
                    {"id": "c1", "tool_name": "echo", "arguments": {"text": "hello"}}
                ],
            ),
            LLMResponse(model="dream", content="done"),
        ]
    )
    harness.register_plugin(plugin)

    agent = AgentConfig(name="t", model="dream", max_steps=4)
    final = await harness.run_session(agent, initial_text="ping")

    # assistant final message persisted
    assert final.last_message is not None and final.last_message.role == Role.ASSISTANT
    assert final.last_message.content == "done"
    # session lifecycle hooks fired
    assert plugin.started == 1 and plugin.finished == 1
    # tool result surfaced into the transcript (schema-driven events)
    assert Role.TOOL in {m.role for m in final.messages}
    # observer saw the tool round-trip and the finish event
    types = {e.type for e in plugin.events}
    assert EventType.TOOL_CALLED.value in types
    assert EventType.TOOL_RESULT.value in types
    assert EventType.ASSISTANT_FINISHED.value in types


async def test_firstresult_provider_selection_with_multiple_plugins() -> None:
    harness = Harness(config=HarnessConfig(auto_load_entry_points=False))
    responder = DreamPlugin(steps=[LLMResponse(model="dream", content="answer")])
    joiner = DreamPlugin(steps=[])  # second plugin only observes
    harness.register_plugin(responder)
    harness.register_plugin(joiner)

    final = await harness.run_session(AgentConfig(name="t"), initial_text="q")
    assert final.last_message is not None and final.last_message.content == "answer"
    # both plugins observed the events
    assert responder.events and joiner.events


async def test_event_bus_ad_hoc_subscription() -> None:
    harness = Harness(config=HarnessConfig(auto_load_entry_points=False))
    seen: list[str] = []
    unsubscribe = harness.bus.on("custom.signal", lambda **kw: seen.append(kw["tag"]))
    await harness.bus.aemit("custom.signal", tag="hello")
    assert seen == ["hello"]
    unsubscribe()
    await harness.bus.aemit("custom.signal", tag="bye")
    assert seen == ["hello"]  # unsubscribed listener ignored


async def test_no_provider_plugin_ends_loop_gracefully() -> None:
    harness = Harness(config=HarnessConfig(auto_load_entry_points=False))
    final = await harness.run_session(AgentConfig(name="bare"), initial_text="x")
    # no agent_next impl -> loop never adds an assistant turn; the user
    # message stays as the last transcript entry
    assert final.session_id is not None
    assert final.last_message is not None and final.last_message.role == Role.USER