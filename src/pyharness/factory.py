"""PyHarness harness factory.

Single source of truth for building a fully-wired Harness instance with:
- OpenRouter HTTP provider or dummy provider
- Complete plugin set: python-exec, fs, web, workflow, subagent, memory, guard-approval, builtin
"""

from __future__ import annotations

import os
from typing import Any

from pyharness import Harness
from pyharness.plugins.guard_approval import ApprovalGuardPlugin
from pyharness.plugins.llm import entry as llm
from pyharness.plugins.llm.http import DEFAULT_BASE
from pyharness.plugins.session_store import SQLiteSessionStorePlugin
from pyharness.plugins.tool_fs import FileSystemPlugin
from pyharness.plugins.tool_python_exec import PythonExecPlugin
from pyharness.plugins.tool_subagent import SubagentToolPlugin
from pyharness.plugins.tool_web import WebPlugin
from pyharness.plugins.workflow import WorkflowPlugin


def build_harness(
    model: str = "dummy",
    provider: str = "dummy",
    api_key: str | None = None,
    base_url: str | None = None,
    verbose: bool = False,
    auto_approve: bool = False,
) -> Harness:
    """Build and return a fully-configured Harness instance.

    Args:
        model: Model name to use (e.g. ``"nvidia/nemotron-3-ultra-550b-a55b:free"``).
        provider: ``"dummy"`` or ``"http"``.
        api_key: API key for HTTP provider. Falls back to ``OPENROUTER_API_KEY`` env var.
        base_url: Base URL for HTTP provider. Falls back to ``OPENROUTER_BASE_URL`` or ``DEFAULT_BASE``.
        verbose: If True, register a CLI observer for event rendering.
        auto_approve: If True, skip the ApprovalGuardPlugin so batch eval runs
            are not blocked by interactive confirmation prompts.

    Returns:
        A configured :class:`~pyharness.core.Harness` ready for ``run_session`` / ``stream_session``.
    """
    harness = Harness()  # auto-loads entry-point plugins (builtin, llm, fs, web, etc.)

    # Register plugin instances (mirrors CLI _harness() wiring)
    harness.register_plugin(FileSystemPlugin())
    harness.register_plugin(WebPlugin())
    harness.register_plugin(WorkflowPlugin())
    harness.register_plugin(SubagentToolPlugin())
    harness.register_plugin(PythonExecPlugin())
    harness.register_plugin(SQLiteSessionStorePlugin())
    if not auto_approve:
        harness.register_plugin(ApprovalGuardPlugin())

    if verbose:
        from rich.console import Console

        _console = Console(highlight=False)

        class _CliObserver:
            @staticmethod
            def observe(context, event):
                if event.type == "tool.called":
                    tool = event.payload.get("tool", "?")
                    _console.print(f"   [dim]tool → {tool}[/]", soft_wrap=True)

        harness.bus.register(_CliObserver())

    # Configure LLM provider
    _configure_provider(harness, provider, model, api_key, base_url)

    harness.initialize()
    return harness


def _configure_provider(
    harness: Harness,
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Wire an LLM provider into the shared registry."""
    llm.clear()

    if provider == "dummy":
        llm.use_dummy(models=(model,))
    elif provider == "http":
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        resolved_base = base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE)
        llm.use_http(models=(model,), api_key=resolved_key, base_url=resolved_base)
    else:
        raise ValueError(f"Unknown provider {provider!r}; use 'dummy' or 'http'")


__all__ = ["build_harness"]
