"""Rich rendering helpers for the PyHarness CLI.

These are pure presentation helpers: they take schema objects and render them.
No engine import — the CLI stays a plugin/consumer over the public Harness API.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pyharness.schema import Message, Role

_ALIASES: dict[str, str] = {
    Role.SYSTEM.value: "system",
    Role.USER.value: "you",
    Role.ASSISTANT.value: "agent",
    Role.TOOL.value: "tool",
}


def transcript_panel(
    console: Console,
    messages: tuple[Message, ...],
    *,
    agent_name: str,
    title: str = "Transcript",
) -> None:
    """Print a framed multi-line transcript."""
    lines: list[Text] = []
    for message in messages:
        label = _ALIASES.get(message.role.value, agent_name)
        text = Text()
        text.append(f"{label}  ", style="bold cyan")
        text.append(message.content or "<empty>")
        # surface tool calls for readability
        for call in message.tool_calls:
            text.append(f"\n   ↳ {call.tool_name}({call.arguments!r})", style="dim")
        lines.append(text)
    body = Text("\n\n").join(lines) if lines else Text("(no messages)")
    console.print(Panel(body, title=title, border_style="blue"))


def session_header(console: Console, **info: Any) -> None:
    """Print a compact key/value banner (agent name, model, session id...)."""
    pairs = Text("  ".join(f"{k}={v}" for k, v in info.items()), style="dim")
    console.print(pairs)