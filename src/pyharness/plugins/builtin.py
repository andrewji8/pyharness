"""Built-in system plugin.

Registered via the ``pyharness.plugins`` entry-point group (see ``pyproject.toml``)
and auto-loaded by :class:`~pyharness.core.Harness`. This is a *pure* plugin: it
owns no engine internals, only the tool it contributes and a few hooks it
observes. It demonstrates the contract every third-party plugin follows.
"""

from __future__ import annotations

import logging

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import Event, ToolArg, ToolResult, ToolSpec

logger = logging.getLogger("pyharness.builtin")

hookimpl = HookimplMarker("pyharness")


def _echo_spec() -> ToolSpec:
    """Declarative spec of the built-in ``echo`` tool."""
    return ToolSpec(
        name="echo",
        description="Echo the given text back. Useful for smoke tests.",
        parameters=(
            ToolArg(name="text", type="string", description="Text to echo", required=True),
        ),
    )


@hookimpl
def get_tool_specs(context: SessionContext) -> tuple[ToolSpec, ...]:
    """Contribute the ``echo`` tool to every session."""
    return (_echo_spec(),)


@hookimpl
async def execute_tool(
    context: SessionContext, tool: ToolSpec, arguments: dict[str, object]
) -> ToolResult | None:
    """Execute a tool this plugin owns; defer everything else (return None)."""
    if tool.name == "echo":
        return ToolResult(
            tool_name=tool.name,
            output={"echo": arguments.get("text", "")},
        )
    return None


@hookimpl
def harness_initialized(harness) -> None:
    """Log once when the fully-wired harness is announced."""
    logger.info("PyHarness builtin plugin active (%s)", harness.config.namespace)


@hookimpl
def observe(context: SessionContext, event: Event) -> None:
    """Best-effort, non-blocking event log (an observer pattern example)."""
    logger.debug("[%s] %s %s", context.session_id.hex[:8], event.type, event.payload)


__all__ = ["execute_tool", "get_tool_specs"]