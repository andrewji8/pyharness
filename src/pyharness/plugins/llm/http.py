"""HTTPProvider — an OpenAI-compatible ``/chat/completions`` client.

Handles both one-shot (``llm_complete``) and SSE streaming (``llm_stream``)
traffic, mapping the provider's wire format onto PyHarness's schema types. Uses
:mod:`httpx` for async transport. Instantiation is lazy: ``httpx`` is imported
inside the network methods, so the package imports fine without it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pyharness.plugins.llm.provider import Provider
from pyharness.schema import LLMRequest, LLMResponse, LLMStreamChunk, ToolCall

# URL is the OpenAI-compatible base (DeepSeek/OpenAI/LM-Studio live here too).
DEFAULT_BASE = "https://api.deepseek.com/v1"


class HTTPProvider(Provider):
    """Chat client for an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        models: tuple[str, ...],
        base_url: str = DEFAULT_BASE,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        transport: Any | None = None,
    ) -> None:
        self._models = models
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra_headers = headers or {}
        self.timeout = timeout
        # Optional httpx.BaseTransport so tests can inject a MockTransport.
        self._transport = transport

    @property
    def name(self) -> str:
        return "http"

    def supported_models(self) -> tuple[str, ...]:
        return self._models

    def supports(self, model: str) -> bool:
        return model in self._models

    # -- internals ---------------------------------------------------------- #
    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _messages_payload(self, request: LLMRequest) -> list[dict[str, Any]]:
        return [{"role": m.role.value, "content": m.content} for m in request.messages]

    @staticmethod
    def _tool_calls(items: list[dict[str, Any]] | None) -> tuple[ToolCall, ...]:
        out: list[ToolCall] = []
        for tc in items or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            out.append(ToolCall(id=tc.get("id"), tool_name=fn.get("name"), arguments=args))
        return tuple(out)

    # -- one-shot ----------------------------------------------------------- #
    async def chat(self, request: LLMRequest) -> LLMResponse | None:
        import httpx

        payload = {
            "model": request.model,
            "messages": self._messages_payload(request),
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._request_headers(),
                json=payload,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        message = data["choices"][0]["message"]
        return LLMResponse(
            model=request.model,
            content=message.get("content") or "",
            tool_calls=self._tool_calls(message.get("tool_calls")),
            usage=data.get("usage", {}),
        )

    # -- streaming ---------------------------------------------------------- #
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        import httpx

        payload = {
            "model": request.model,
            "messages": self._messages_payload(request),
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        assembler = _StreamToolAssembler()
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._request_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for raw in response.aiter_lines():
                    if not raw.startswith("data:"):
                        continue
                    datum = raw[5:].strip()
                    if not datum or datum == "[DONE]":
                        break
                    obj = json.loads(datum)
                    delta = obj["choices"][0]["delta"]
                    delta_content = delta.get("content")
                    if delta_content:
                        yield LLMStreamChunk(delta=delta_content)
                    for call in assembler.take(delta.get("tool_calls")):
                        yield LLMStreamChunk(delta="", tool_calls=(call,))


class _StreamToolAssembler:
    """Reassemble split function-call deltas (SSE) into whole ToolCalls.

    Providers send tool-arguments in fragments across multiple chunks, keyed by
    the same ``index``. This buffer glues fragments per index and yields a
    complete :class:`ToolCall` once its arguments finish.
    """

    def __init__(self) -> None:
        self._parts: dict[int, dict[str, Any]] = {}

    def take(self, deltas: list[dict[str, Any]] | None) -> list[ToolCall]:
        if not deltas:
            return []
        complete: list[ToolCall] = []
        for piece in deltas:
            index = piece.get("index", 0)
            slot = self._parts.setdefault(index, {"id": "", "name": "", "args": []})
            function = piece.get("function", {})
            if function.get("name"):
                slot["name"] = function["name"]
            if piece.get("id"):
                slot["id"] = piece["id"]
            if function.get("arguments"):
                slot["args"].append(function["arguments"])
                if not _args_open(slot["args"]):
                    try:
                        arguments = json.loads("".join(slot["args"]))
                    except json.JSONDecodeError:
                        arguments = {}
                    complete.append(
                        ToolCall(id=slot["id"] or "", tool_name=slot["name"], arguments=arguments)
                    )
                    del self._parts[index]
        return complete


def _args_open(fragments: list[str]) -> bool:
    """Heuristic: args remain 'open' while their JSON braces are unmatched."""
    joined = "".join(fragments)
    return joined.count("{") - joined.count("}") > 0


__all__ = ["DEFAULT_BASE", "HTTPProvider"]