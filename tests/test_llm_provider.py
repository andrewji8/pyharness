"""Tests for module A — the LLM provider plugin.

Validates the engine↔provider contract end to end:

* engine assembles a schema-driven ``LLMRequest`` from the transcript;
* ``llm_complete`` (one-shot) and ``llm_stream`` (async generator) hooks drive
  ``run_session`` / ``stream_session``;
* provider selection/fail-over is by model name, engine-side (first non-None);
* the streaming loop still executes tools and persists assistant messages.
"""

from __future__ import annotations

import asyncio

from pluggy import HookimplMarker

from pyharness import Harness
from pyharness.context import open_session
from pyharness.plugins.llm import entry as llm
from pyharness.schema import (
    AgentConfig,
    HarnessConfig,
    LLMResponse,
    Message,
    Role,
    ToolArg,
    ToolCall,
    ToolResult,
    ToolSpec,
)

hookimpl = HookimplMarker("pyharness")

ECHO_SPEC = ToolSpec(
    name="echo",
    description="Echo text.",
    parameters=(ToolArg(name="text", type="string", required=True),),
)


class EchoToolPlugin:
    """A minimal tool plugin so streaming tool-call loops have a sink."""

    @hookimpl
    def get_tool_specs(self, context) -> tuple[ToolSpec, ...]:
        return (ECHO_SPEC,)

    @hookimpl
    async def execute_tool(self, context, tool, arguments) -> ToolResult | None:
        if tool.name == "echo":
            return ToolResult(tool_name=tool.name, output={"echo": arguments.get("text", "")})
        return None


def _harness(*plugins: object, auto_load: bool = False) -> Harness:
    """Harness with the LLM plugin + extra plugins wired in (no auto-load)."""
    h = Harness(config=HarnessConfig(auto_load_entry_points=auto_load))
    h.register_plugin(llm)  # the LLM plugin's hookimpls
    for plugin in plugins:
        h.register_plugin(plugin)
    return h


def _msg(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


# --------------------------------------------------------------------------- #
# Request assembly
# --------------------------------------------------------------------------- #
def test_build_request_prepends_system_prompt() -> None:
    h = _harness()
    agent = AgentConfig(name="a", model="dummy", system_prompt="you are terse")

    async def without_user() -> None:
        async with open_session() as ctx:
            request = await h.build_request(ctx, agent)
            assert [m.role for m in request.messages] == [Role.SYSTEM]
            assert request.messages[0].content == "you are terse"

    async def with_user() -> None:
        async with open_session() as ctx:
            ctx = ctx.append_message(_msg(Role.USER, "hi"))
            request = await h.build_request(ctx, agent)
            assert [m.role for m in request.messages] == [Role.SYSTEM, Role.USER]

    asyncio.run(without_user())
    asyncio.run(with_user())


# --------------------------------------------------------------------------- #
# One-shot completion (llm_complete)
# --------------------------------------------------------------------------- #
async def test_llm_complete_via_run_session() -> None:
    llm.clear()
    llm.use_dummy(models=("dummy",), plan=[LLMResponse(model="dummy", content="hi there")])
    h = _harness()

    final = await h.run_session(AgentConfig(name="a", model="dummy"), initial_text="hello")
    assert final.last_message is not None
    assert final.last_message.role == Role.ASSISTANT
    assert final.last_message.content == "hi there"


async def test_provider_falls_through_when_model_unknown() -> None:
    llm.clear()
    llm.use_dummy(models=("known",), plan=[LLMResponse(model="known", content="ok")])
    h = _harness()

    final = await h.run_session(AgentConfig(name="a", model="unknown"), initial_text="x")
    # no provider serves "unknown" -> no assistant turn; user msg persists
    assert final.last_message is not None and final.last_message.role == Role.USER


async def test_get_llm_providers_lists_models() -> None:
    llm.clear()
    llm.use_dummy(models=("alpha", "beta"))
    h = _harness()
    async with open_session() as ctx:
        # get_llm_providers is a fan-in hook: each plugin contributes a tuple,
        # so flatten across the (single) relay result.
        names: set[str] = set()
        for group in h.bus.pm.hook.get_llm_providers(context=ctx):
            names.update(group)
    assert {"alpha", "beta"} <= names


# --------------------------------------------------------------------------- #
# Streaming (llm_stream) + tools
# --------------------------------------------------------------------------- #
async def test_stream_session_yields_deltas_and_sets_last_context() -> None:
    llm.clear()
    llm.use_dummy(models=("dummy",), plan=[LLMResponse(model="dummy", content="helloworld")], chunk_size=3)
    h = _harness()

    chunks = [c async for c in h.stream_session(AgentConfig(name="a", model="dummy"), initial_text="go")]
    deltas = "".join(c.delta for c in chunks)
    assert deltas == "helloworld"
    assert h.last_context is not None
    assert h.last_context.last_message is not None
    assert h.last_context.last_message.role == Role.ASSISTANT
    assert h.last_context.last_message.content == "helloworld"


async def test_stream_session_runs_tools_in_the_loop() -> None:
    llm.clear()
    plan = [
        LLMResponse(
            model="dummy",
            content="",
            tool_calls=(ToolCall(id="c1", tool_name="echo", arguments={"text": "yo"}),),
        ),
        LLMResponse(model="dummy", content="done"),
    ]
    llm.use_dummy(models=("dummy",), plan=plan, chunk_size=2)
    h = _harness(EchoToolPlugin())

    collected = [c async for c in h.stream_session(AgentConfig(name="a", model="dummy"), initial_text="ping")]
    # em-dash: the assistant turn "done" streamed; tool message appended silently
    assert "".join(c.delta for c in collected).endswith("done")
    assert h.last_context is not None
    roles = [m.role for m in h.last_context.messages]
    assert Role.TOOL in roles
    assert h.last_context.last_message is not None
    assert h.last_context.last_message.role == Role.ASSISTANT
    assert h.last_context.last_message.content == "done"


async def test_dummy_provider_streams_tool_calls_trailing_chunk() -> None:
    from pyharness.schema import LLMRequest

    llm.clear()
    provider = llm.use_dummy(
        models=("dummy",),
        plan=[
            LLMResponse(
                model="dummy",
                content="",
                tool_calls=(ToolCall(id="c1", tool_name="echo", arguments={"text": "z"}),),
            )
        ],
    )
    request = LLMRequest(model="dummy", messages=())
    chunks = [c for c in await _collect(provider.stream(request))]
    assert any(c.tool_calls for c in chunks)


async def test_stream_session_continues_on_reversible_lineage() -> None:
    llm.clear()
    llm.use_dummy(
        models=("dummy",),
        plan=[
            LLMResponse(model="dummy", content="first reply"),
            LLMResponse(model="dummy", content="second reply"),
        ],
        chunk_size=6,
    )
    h = _harness()

    agent = AgentConfig(name="a", model="dummy")
    first = [c async for c in h.stream_session(agent, initial_text="Q1")]
    assert "".join(c.delta for c in first) == "first reply"
    assert h.last_context is not None
    assert len(h.last_context.messages) == 2  # user + assistant

    # Second turn continues the SAME session_id/lineage (no fresh context).
    second = [c async for c in h.stream_session(agent, initial_text="Q2", continue_from=h.last_context)]
    assert "".join(c.delta for c in second) == "second reply"
    assert h.last_context is not None
    context = h.last_context
    assert len(context.messages) == 4  # 2 + user + assistant
    roles = [m.role for m in context.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]


async def test_run_session_executes_tool_calls_and_appends_tool_message() -> None:
    """Full chain: LLM returns tool_calls -> engine executes tool -> result appended."""
    llm.clear()
    plan = [
        LLMResponse(
            model="dummy",
            content="",
            tool_calls=(ToolCall(id="c1", tool_name="echo", arguments={"text": "tool-result"}),),
        ),
        LLMResponse(model="dummy", content="final reply"),
    ]
    llm.use_dummy(models=("dummy",), plan=plan, chunk_size=2)
    h = _harness(EchoToolPlugin())

    await h.run_session(AgentConfig(name="a", model="dummy"), initial_text="ping")

    assert h.last_context is not None
    roles = [m.role for m in h.last_context.messages]
    assert Role.TOOL in roles, f"Expected TOOL message in {roles}"
    tool_msgs = [m for m in h.last_context.messages if m.role == Role.TOOL]
    assert len(tool_msgs) == 1
    assert "tool-result" in tool_msgs[0].content
    assert h.last_context.last_message is not None
    assert h.last_context.last_message.role == Role.ASSISTANT
    assert h.last_context.last_message.content == "final reply"


async def _collect(agen):
    return [c async for c in agen]