"""DummyProvider — a deterministic, network-free provider.

Used for offline smoke tests, CI, and rapid UI development when no API key is
present. You hand it a *plan* of :class:`LLMResponse` objects (you can craft
tool-calling turns) and it serves them in order, optionally splitting each
assistant turn into small :class:`LLMStreamChunk` deltas for streaming tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pyharness.plugins.llm.provider import Provider
from pyharness.schema import LLMRequest, LLMResponse, LLMStreamChunk


class DummyProvider(Provider):
    """Serves a scripted plan of responses for a set of model names."""

    def __init__(
        self,
        *,
        models: tuple[str, ...] = ("dummy",),
        plan: list[LLMResponse] | None = None,
        default: str = "",
        chunk_size: int = 3,
        delay: float = 0.0,
    ) -> None:
        self._models = models
        self._plan = list(plan or [LLMResponse(model=models[0], content=default)])
        self._chunk_size = max(1, chunk_size)
        self._delay = delay

    @property
    def name(self) -> str:
        return "dummy"

    def supported_models(self) -> tuple[str, ...]:
        return self._models

    def supports(self, model: str) -> bool:
        return "*" in self._models or model in self._models

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Pop the next scripted response (or a scripted default)."""
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._plan:
            return self._plan.pop(0)
        return LLMResponse(model=request.model, content=self._plan_default(self._models[0]))

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Stream the next scripted response as small deltas, then any tool
        calls as a trailing chunk."""
        resp = await self.chat(request)
        content = resp.content or ""
        for i in range(0, len(content), self._chunk_size):
            await asyncio.sleep(self._delay)
            yield LLMStreamChunk(delta=content[i : i + self._chunk_size])
        if resp.tool_calls:
            yield LLMStreamChunk(delta="", tool_calls=resp.tool_calls)

    def _plan_default(self, model: str) -> str:
        return f"[dummy:{model}] no more scripted responses"