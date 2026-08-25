"""HTTPProvider — an OpenAI-compatible ``/chat/completions`` client.

Handles both one-shot (``llm_complete``) and SSE streaming (``llm_stream``)
traffic, mapping the provider's wire format onto PyHarness's schema types. Uses
:mod:`httpx` for async transport. Instantiation is lazy: ``httpx`` is imported
inside the network methods, so the package imports fine without it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger("pyharness.llm.http")

_LLM_DIRECT = os.environ.get("LLM_DIRECT") == "1"

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

    @staticmethod
    def _normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
        """Keep only scalar integer fields from provider usage payloads."""
        normalized: dict[str, int] = {}
        for key, value in (usage or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                normalized[key] = value
        return normalized

    def _messages_payload(self, request: LLMRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for m in request.messages:
            msg: dict[str, Any] = {"role": m.role.value, "content": m.content}
            if m.name:
                msg["name"] = m.name
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments) if not isinstance(tc.arguments, str) else tc.arguments,
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            messages.append(msg)
        return messages

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
        if request.tools:
            payload["tools"] = request.tools

        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self._transport,
            trust_env=not _LLM_DIRECT,
        ) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._request_headers(),
                json=payload,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        choices = data.get("choices") or []
        if not choices:
            return LLMResponse(model=request.model, content="", tool_calls=(), usage={})
        message = choices[0].get("message", {})
        return LLMResponse(
            model=request.model,
            content=message.get("content") or "",
            tool_calls=self._tool_calls(message.get("tool_calls")),
            usage=self._normalize_usage(data.get("usage", {})),
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
        if request.tools:
            payload["tools"] = request.tools

        assembler = _StreamToolAssembler()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    transport=self._transport,
                    trust_env=not _LLM_DIRECT,
                ) as client:
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
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            delta_content = delta.get("content")
                            if delta_content:
                                yield LLMStreamChunk(delta=delta_content)
                            for call in assembler.take(delta.get("tool_calls")):
                                yield LLMStreamChunk(delta="", tool_calls=(call,))
                return
            except (
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.ConnectError,
                httpx.ReadTimeout,
            ) as exc:
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning("stream attempt %d failed: %s, retrying in %ds", attempt + 1, exc, backoff)
                    await asyncio.sleep(backoff)
                    continue
                logger.error("stream failed after %d attempts: %s", max_retries, exc)
                raise


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