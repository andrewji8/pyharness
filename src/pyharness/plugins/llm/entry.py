"""LLM provider plugin — the "everything is a plugin" LLM surface.

Registered via the ``pyharness.plugins`` entry-point group (``.llm``). This module
exposes module-level ``@hookimpl`` functions in front of a :class:`ProviderRegistry`.
Providers are registered by importing this module and calling ``register_provider``
/ ``use_dummy`` / ``use_http``; a minimal env-var convention auto-registers a real
HTTP provider on import. The engine talks to this plugin only through the
``llm_complete`` / ``llm_stream`` / ``get_llm_providers`` hooks.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.plugins.llm.dummy import DummyProvider
from pyharness.plugins.llm.http import DEFAULT_BASE, HTTPProvider
from pyharness.plugins.llm.provider import Provider, ProviderRegistry
from pyharness.schema import LLMRequest, LLMResponse, LLMStreamChunk

hookimpl = HookimplMarker("pyharness")

# Canonical runtime registry for this plugin instance.
_registry: ProviderRegistry = ProviderRegistry()

# Env-var convention used to seed a real HTTP provider without code.
_ENV_DEFAULTS = {
    "DEEPSEEK": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", ("deepseek-chat", "deepseek-reasoner")),
    "OPENAI": ("OPENAI_API_KEY", "https://api.openai.com/v1", ("gpt-4o-mini", "gpt-4o")),
    "LLAMA": ("OLLAMA_API_KEY", "http://localhost:11434/v1", ("llama3.1",)),
}


def clear() -> None:
    """Drop all registered providers (helpful in tests)."""
    _registry.clear()


def register_provider(provider: Provider) -> None:
    """Add a provider instance to the plugin's registry."""
    _registry.add(provider)


def use_dummy(*, models: tuple[str, ...] = ("dummy",), plan: list[LLMResponse] | None = None, **kwargs: Any) -> DummyProvider:
    """Convenience: build and register a network-free :class:`DummyProvider`."""
    provider = DummyProvider(models=models, plan=plan, **kwargs)
    register_provider(provider)
    return provider


def use_http(
    *,
    models: tuple[str, ...],
    base_url: str | None = None,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> HTTPProvider:
    """Convenience: build and register a real :class:`HTTPProvider`."""
    provider = HTTPProvider(
        models=models,
        base_url=base_url or DEFAULT_BASE,
        api_key=api_key,
        headers=headers,
        timeout=timeout,
    )
    register_provider(provider)
    return provider


def auto_configure_from_env() -> None:
    """Seed providers from ``*_API_KEY`` style env vars (idempotent)."""
    for label, (key_env, base_url, models) in _ENV_DEFAULTS.items():
        if os.environ.get(key_env) and not _registry.get(models[0]):
            use_http(models=models, base_url=base_url, api_key=os.environ[key_env])


# --------------------------------------------------------------------------- #
# Hook implementation (the plugin's public contract)
# --------------------------------------------------------------------------- #
@hookimpl
def get_llm_providers(context: SessionContext) -> tuple[str, ...]:
    """Expose serveable model names for negotiation/UIs."""
    return _registry.models()


@hookimpl
async def llm_complete(context: SessionContext, request: LLMRequest) -> LLMResponse | None:
    """Complete a request through the registry (or None to defer)."""
    return await _registry.chat(request)


@hookimpl
async def llm_stream(context: SessionContext, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
    """Stream deltas through the registry (async generator hook)."""
    provider = _registry.get(request.model)
    if provider is None:
        return
    async for chunk in provider.stream(request):
        yield chunk


# Auto-configure from environment when imported (safe: no-op unless key present).
auto_configure_from_env()

__all__ = [
    "auto_configure_from_env",
    "clear",
    "get_llm_providers",
    "hookimpl",
    "llm_complete",
    "llm_stream",
    "register_provider",
    "use_dummy",
    "use_http",
]