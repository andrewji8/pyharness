"""PyHarness CLI — `dsh-py`.

Built as a **plugin** and consumer of the public Harness API:

* it opens a ``Harness`` (which auto-loads the ``builtin``/``llm`` plugins),
* it configures an LLM provider via the LLM plugin's registry,
* when ``--verbose`` it registers its own ``CliObserver`` ``@hookimpl`` to render
  the event stream — i.e. the CLI UI is itself a plugin in the harness it owns.

Commands: ``run`` (one-shot or multi-turn, rendered as a rich transcript),
``chat`` (streaming REPL), ``version``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import typer
from pluggy import HookimplMarker
from rich.console import Console

from pyharness import Harness
from pyharness.context import SessionContext
from pyharness.plugins.cli.ui import session_header, transcript_panel
from pyharness.plugins.llm import entry as llm
from pyharness.plugins.llm.http import DEFAULT_BASE
from pyharness.schema import AgentConfig, Event, LLMResponse

hookimpl = HookimplMarker("pyharness")
console = Console(highlight=False)

app = typer.Typer(
    help="PyHarness — everything is a plugin. CLI for agent sessions.",
    no_args_is_help=True,
)

_EXITS = {"/exit", "/quit", "exit", "quit", "/q"}


class CliObserver:
    """A tiny plugin that renders harness events to the terminal (--verbose)."""

    @hookimpl
    def observe(self, context: SessionContext, event: Event) -> None:
        if event.type == "tool.called":
            tool = event.payload.get("tool", "?")
            console.print(f"   [dim]tool → {tool}[/]", soft_wrap=True)


def _configure_provider(
    provider: str,
    models: tuple[str, ...],
    api_key: str | None,
    base_url: str | None,
    replies: tuple[str, ...] = (),
) -> None:
    """Wire a provider into the shared LLM-plugin registry."""
    llm.clear()
    if provider == "dummy":
        plan = [LLMResponse(model=models[0], content=text) for text in replies]
        if not plan:
            plan = [LLMResponse(model=models[0], content="PyHarness reply (dummy provider).")]
        llm.use_dummy(models=models, plan=plan)
    elif provider == "http":
        llm.use_http(models=models, api_key=api_key, base_url=base_url or DEFAULT_BASE)
    else:
        raise typer.BadParameter(f"unknown provider {provider!r}; use 'dummy' or 'http'")


def _harness(verbose: bool) -> Harness:
    harness = Harness()  # auto-loads builtin + llm entry-points
    if verbose:
        harness.bus.register(CliObserver())
    return harness


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
@app.command("version")
def version() -> None:
    """Print the installed version."""
    from pyharness import __version__

    console.print(f"PyHarness {__version__}")


@app.command("run")
def run(
    model: str = typer.Option("dummy", "--model", "-m", help="Model to call."),
    name: str = typer.Option("cli-agent", "--name", help="Agent display name."),
    provider: str = typer.Option("dummy", "--provider", "-p", help="'dummy' or 'http'."),
    system_prompt: str = typer.Option("", "--system-prompt", help="System prompt."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key (http only)."),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible base (http only)."),
    initial: str = typer.Option("Hello!", "--initial", "-i", help="First user turn."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Render harness events."),
) -> None:
    """Run one user turn against an agent and print the transcript."""
    _configure_provider(provider, (model,), api_key, base_url, replies=(initial,))
    harness = _harness(verbose)
    agent = AgentConfig(name=name, model=model, system_prompt=system_prompt)
    session_header(console, agent=name, model=model, provider=provider)

    state = asyncio.run(_run_turn(harness, agent, initial, None))

    title = f"Session {state.session_id.hex[:8] if state else ''}".strip()
    transcript_panel(console, state.messages if state else (), agent_name=name, title=title)


async def _run_turn(
    harness: Harness, agent: AgentConfig, text: str, state: SessionContext | None
) -> SessionContext:
    console.print(f"[bold cyan]you[/]  {text}")
    await harness.run_session(agent, initial_text=text, continue_from=state)
    return harness.last_context if harness.last_context is not None else state


@app.command("chat")
def chat(
    model: str = typer.Option("dummy", "--model", "-m", help="Model to call."),
    name: str = typer.Option("cli-agent", "--name", help="Agent display name."),
    provider: str = typer.Option("dummy", "--provider", "-p", help="'dummy' or 'http'."),
    system_prompt: str = typer.Option("", "--system-prompt", help="System prompt."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key (http only)."),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible base (http only)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Render harness events."),
) -> None:
    """Streaming REPL. `/new` resets context; `/exit` quits."""
    _configure_provider(provider, (model,), api_key, base_url)
    harness = _harness(verbose)
    agent = AgentConfig(name=name, model=model, system_prompt=system_prompt)
    console.print(f"[green]PyHarness chat[/] (model={model}) — /new to reset, /exit to quit.")

    state: SessionContext | None = None
    while True:
        try:
            text: str = console.input("[bold cyan]you> [/]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        stripped = text.strip()
        if stripped.lower() in _EXITS:
            break
        if stripped == "/new":
            state = None
            continue
        state = asyncio.run(_stream_turn(harness, agent, text, state))


async def _stream_turn(
    harness: Harness, agent: AgentConfig, text: str, state: SessionContext | None
) -> SessionContext:
    console.print(f"[bold green]{agent.name}> [/]", end="")
    async for chunk in harness.stream_session(agent, initial_text=text, continue_from=state):
        console.print(chunk.delta, end="", soft_wrap=True)
    console.print()
    return harness.last_context if harness.last_context is not None else state


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(3080, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code change."),
) -> None:
    """Start the PyHarness Web UI (FastAPI + WebSocket)."""
    import uvicorn

    from pyharness.plugins.web_ui import serve as serve_web_ui

    harness = _harness(verbose=False)
    serve_web_ui(harness, host=host, port=port)


main = app  # console-script entry point (`dsh-py`)

__all__ = ["app", "main"]