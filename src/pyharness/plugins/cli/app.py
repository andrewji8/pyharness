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
import base64
from typing import Any

import typer
from rich.console import Console

from pyharness.context import SessionContext
from pyharness.factory import build_harness
from pyharness.plugins.cli.ui import session_header, transcript_panel
from pyharness.plugins.tool_python_exec import _DOCKER_IMAGE
from pyharness.schema import AgentConfig, ContentPart, Message, Role
from pyharness import Harness
from pathlib import Path

console = Console(highlight=False)

app = typer.Typer(
    help="PyHarness — everything is a plugin. CLI for agent sessions.",
    no_args_is_help=True,
)

_EXITS = {"/exit", "/quit", "exit", "quit", "/q"}


def _harness(verbose: bool, model: str = "dummy", provider: str = "dummy", api_key: str | None = None, base_url: str | None = None) -> Harness:
    try:
        return build_harness(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            verbose=verbose,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


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
    harness = _harness(verbose, model=model, provider=provider, api_key=api_key, base_url=base_url)
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
    image: list[str] | None = typer.Option(None, "--image", help="Attach image file(s) as image parts (repeatable)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Render harness events."),
) -> None:
    """Streaming REPL. `/new` resets context; `/exit` quits."""
    harness = _harness(verbose, model=model, provider=provider, api_key=api_key, base_url=base_url)
    agent = AgentConfig(name=name, model=model, system_prompt=system_prompt)
    console.print(f"[green]PyHarness chat[/] (model={model}) — /new to reset, /exit to quit.")

    image_parts = _build_image_parts(image) if image else []

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
        state = asyncio.run(_stream_turn(harness, agent, text, state, image_parts))


async def _stream_turn(
    harness: Harness,
    agent: AgentConfig,
    text: str,
    state: SessionContext | None,
    parts: list[ContentPart] | None = None,
) -> SessionContext:
    console.print(f"[bold green]{agent.name}> [/]", end="")
    if parts:
        message = Message(role=Role.USER, content=text, parts=tuple(parts))
        stream = harness.stream_session(agent, initial_message=message, continue_from=state)
    else:
        stream = harness.stream_session(agent, initial_text=text, continue_from=state)
    async for chunk in stream:
        console.print(chunk.delta, end="", soft_wrap=True)
    console.print()
    return harness.last_context if harness.last_context is not None else state


_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _build_image_parts(paths: list[str]) -> list[ContentPart]:
    """Read local image files and encode as base64 data-URL image parts."""
    parts: list[ContentPart] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise typer.BadParameter(f"图片文件不存在: {raw}")
        mime = _IMAGE_MIME.get(path.suffix.lower(), "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode()
        parts.append(ContentPart(type="image", url=f"data:{mime};base64,{b64}"))
    return parts


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


@app.command("worker")
def worker(
    poll_timeout: float = typer.Option(1.0, "--poll", help="BRPOP poll timeout (s)."),
) -> None:
    """Run a distributed-execution worker consuming the Redis task queue.

    Requires ``REDIS_URL`` to be set. Consumes ``ToolTask`` messages, executes
    them locally, and streams results + ``tool.stream`` events back to web nodes.
    """
    from pyharness.plugins.worker import run_worker_forever

    console.print("[cyan]Starting distributed worker... (Ctrl-C to stop)[/]")
    try:
        asyncio.run(run_worker_forever(poll_timeout=poll_timeout))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("[dim]worker stopped[/]")


@app.command("sandbox")
def sandbox(
    action: str = typer.Argument(..., help="Action: init"),
) -> None:
    """Manage the python_exec Docker sandbox."""
    if action != "init":
        raise typer.BadParameter(f"Unknown sandbox action: {action!r}; use 'init'")

    from pyharness.plugins.tool_python_exec import PythonExecPlugin

    plugin = PythonExecPlugin()
    console.print("[cyan]Initializing Docker sandbox...[/]")
    if asyncio.run(plugin.init_sandbox()):
        console.print("[green]Sandbox initialized successfully.[/]")
        console.print(f"Image: {_DOCKER_IMAGE}")
    else:
        console.print("[red]Failed to initialize sandbox. Is Docker installed and running?[/]")
        raise typer.Exit(1)


def _find_evals_dir() -> Path:
    for base in Path(__file__).resolve().parents:
        if (base / "evals").is_dir():
            return base / "evals"
    return Path.cwd() / "evals"


@app.command("eval")
def eval(
    suite: str = typer.Option("basic", "--suite", "-s", help="Suite name (e.g. basic)."),
    model: str = typer.Option("dummy", "--model", "-m", help="Model name for eval tasks."),
    judge: str | None = typer.Option(None, "--judge", "-j", help="Judge model name."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM judge, use programmatic only."),
) -> None:
    """Run the built-in evaluation suite."""
    from pyharness.plugins.eval_runner import run_evals

    suite_path = _find_evals_dir() / f"{suite}.yaml"
    if not suite_path.exists():
        console.print(f"[red]Suite not found: {suite_path}[/]")
        raise typer.Exit(1)

    console.print(f"[cyan]Running eval suite: {suite}[/]")
    if no_judge:
        console.print("[yellow]LLM judge disabled[/]")

    report = asyncio.run(run_evals(suite_path=suite_path, judge_model=judge, model=model))
    if report.failed_count > 0:
        raise typer.Exit(1)


main = app  # console-script entry point (`dsh-py`)

__all__ = ["app", "main"]


# --------------------------------------------------------------------------- #
# Plugin lifecycle (hot-reload) — talks to a running `pyharness serve`
# --------------------------------------------------------------------------- #
@app.command("plugin")
def plugin(
    action: str = typer.Argument(..., help="list | load | unload | reload"),
    target: str = typer.Argument(None, help="path (load) or name (unload/reload)"),
    url: str = typer.Option("http://127.0.0.1:3080", "--url", help="Running serve URL"),
) -> None:
    """Hot-load / unload / reload plugins on a running server.

    Examples::

        pyharness plugin list
        pyharness plugin load ./my_plugin.py
        pyharness plugin reload my_plugin
        pyharness plugin unload my_plugin
    """
    import httpx

    endpoint = f"{url.rstrip('/')}/api/plugins"
    try:
        # CLI only talks to a local `pyharness serve`; bypass system proxy.
        with httpx.Client(trust_env=False, timeout=30) as client:
            if action == "list":
                resp = client.get(endpoint, timeout=10)
            elif action == "load":
                if not target:
                    raise typer.BadParameter("load 需要一个插件文件路径")
                resp = client.post(endpoint + "/load", json={"path": target}, timeout=30)
            elif action == "reload":
                if not target:
                    raise typer.BadParameter("reload 需要一个插件名称")
                resp = client.post(endpoint + "/reload", json={"name": target}, timeout=30)
            elif action == "unload":
                if not target:
                    raise typer.BadParameter("unload 需要一个插件名称")
                resp = client.request("DELETE", endpoint + f"/{target}", timeout=30)
            else:
                raise typer.BadParameter(f"未知 action: {action!r} (list|load|unload|reload)")
    except httpx.HTTPError as exc:
        console.print(f"[red]无法连接 {url}: {exc}[/]")
        raise typer.Exit(1)

    try:
        payload = resp.json()
    except ValueError:
        console.print(f"[red]服务器返回非 JSON ({resp.status_code}):[/]\n{resp.text}")
        raise typer.Exit(1)

    if isinstance(payload, dict) and payload.get("ok") is False:
        console.print(f"[red]操作失败:[/] {payload.get('error', '未知错误')}")
        raise typer.Exit(1)
    if isinstance(payload, dict) and "error" in payload:
        console.print(f"[red]错误:[/] {payload['error']}")
        raise typer.Exit(1)

    console.print_json(resp.text)