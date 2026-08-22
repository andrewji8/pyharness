"""Provider abstraction for the LLM plugin.

A *provider* is a concrete connection to one LLM API family (a transport). The
schema-driven :class:`LLMRequest`/``LLMResponse`` types are the only vocabulary
a provider speaks — no provider imports engine internals. Provilers are
discovered by ``model`` name (``supports``), which lets several providers coexist
and lets a request fail-over to the next registered provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pyharness.schema import LLMRequest, LLMResponse, LLMStreamChunk


class Provider(ABC):
    """A transport to an LLM API. Implement :meth:`chat` and, optionally,
    :meth:`stream`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider identity (also used for logging/metrics)."""

    @abstractmethod
    def supports(self, model: str) -> bool:
        """Return True if this provider can serve ``model``."""

    def supported_models(self) -> tuple[str, ...]:
        """Every model this provider can serve (for capability disclosure)."""
        return ()

    @abstractmethod
    async def chat(self, request: LLMRequest) -> LLMResponse | None:
        """Complete ``request`` in one shot. ``None`` means "not handled here"
        so the request can fail over to another provider."""

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Stream deltas for ``request``. Defaults to no streaming; providers
        that support it should override and ``yield`` :class:`LLMStreamChunk`."""
        if False:  # pragma: no cover - makes this an (empty) async generator
            yield LLMStreamChunk(delta="")


class ProviderRegistry:
    """Ordered collection of providers; resolution is first-``supports`` wins."""

    def __init__(self, providers: Any = ()) -> None:
        self._providers: list[Provider] = list(providers)

    def add(self, provider: Provider) -> None:
        """Register a provider (idempotent by identity)."""
        if provider not in self._providers:
            self._providers.append(provider)

    def remove(self, name: str) -> None:
        """Remove the first provider whose ``name`` matches (if any)."""
        for provider in self._providers:
            if provider.name == name:
                self._providers.remove(provider)
                break

    def clear(self) -> None:
        """Drop all registered providers."""
        self._providers.clear()

    def models(self) -> tuple[str, ...]:
        """Names of every model serveable by the current provider set, in
        deterministic registration order (never relies on ``set`` hashing)."""
        seen: list[str] = []
        for provider in self._providers:
            for model in provider.supported_models():
                if model not in seen:
                    seen.append(model)
        return tuple(seen)

    def get(self, model: str) -> Provider | None:
        """First provider able to serve ``model``, else None."""
        for provider in self._providers:
            if provider.supports(model):
                return provider
        return None

    async def chat(self, request: LLMRequest) -> LLMResponse | None:
        """Complete ``request`` through the first-``supports`` provider."""
        provider = self.get(request.model)
        if provider is None:
            return None
        return await provider.chat(request)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Stream ``request`` through the first-``supports`` provider."""
        provider = self.get(request.model)
        if provider is None:
            return empty_stream()
        return provider.stream(request)


async def empty_stream() -> AsyncIterator[LLMStreamChunk]:
    """Trivial async generator used when no provider can stream a model."""
    if False:  # pragma: no cover
        yield LLMStreamChunk(delta="")


__all__ = ["Provider", "ProviderRegistry", "empty_stream"]