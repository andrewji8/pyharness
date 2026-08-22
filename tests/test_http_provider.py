"""Transport-level tests for the HTTPProvider (OpenAI-compatible).

Uses ``httpx.MockTransport`` so no network is required. Proves the real provider
correctly maps wire payloads onto the schema types: one-shot completions, SSE
streaming, and reassembly of fragmented tool-call deltas.
"""

from __future__ import annotations

import json

import httpx

from pyharness.plugins.llm.http import HTTPProvider
from pyharness.schema import LLMRequest, Message, Role

MODELS = ("test-model",)


async def test_chat_one_shot_maps_payload_and_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert request.headers["Authorization"] == "Bearer sekrit"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello world", "tool_calls": None}}],
                "usage": {"total_tokens": 7},
            },
        )

    provider = HTTPProvider(
        models=MODELS,
        base_url="http://test/v1",
        api_key="sekrit",
        transport=httpx.MockTransport(handler),
    )
    resp = await provider.chat(
        LLMRequest(model="test-model", messages=(Message(role=Role.USER, content="hi"),))
    )
    assert resp is not None
    assert resp.content == "hello world"
    assert resp.usage == {"total_tokens": 7}


async def test_stream_reassembles_sse_deltas() -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"foo"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"bar"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=sse)

    provider = HTTPProvider(
        models=MODELS, base_url="http://test", transport=httpx.MockTransport(handler)
    )
    chunks = [c async for c in provider.stream(LLMRequest(model="test-model", messages=()))]
    assert "".join(c.delta for c in chunks) == "foobar"


async def test_stream_reassembles_split_tool_call() -> None:
    sse = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function",'
        '"function":{"name":"echo","arguments":"{\\"tex"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"t\\":\\"yo\\"}"}}]}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse)

    provider = HTTPProvider(
        models=MODELS, base_url="http://test", transport=httpx.MockTransport(handler)
    )
    calls = []
    async for chunk in provider.stream(LLMRequest(model="test-model", messages=())):
        calls.extend(chunk.tool_calls)
    assert len(calls) == 1
    assert calls[0].tool_name == "echo"
    assert calls[0].arguments == {"text": "yo"}


async def test_registry_defers_unsupported_model() -> None:
    from pyharness.plugins.llm.provider import ProviderRegistry

    def boom_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("provider must not be called for an unsupported model")

    provider = HTTPProvider(
        models=MODELS, base_url="http://test", transport=httpx.MockTransport(boom_handler)
    )
    assert provider.supports("test-model")
    assert not provider.supports("elsewhere")

    registry = ProviderRegistry((provider,))
    resp = await registry.chat(LLMRequest(model="elsewhere", messages=()))
    assert resp is None