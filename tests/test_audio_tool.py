"""Tests for the audio transcription tool (batch 2/3)."""

from __future__ import annotations

import base64

import httpx
import pytest

from pyharness.context import SessionContext
from pyharness.plugins import tool_audio
from pyharness.plugins.tool_audio import AudioToolPlugin, transcribe_bytes


def _mock_openai_client(handler) -> type:
    """Build a stand-in for ``httpx.AsyncClient`` backed by a MockTransport."""
    # Capture the real client class: monkeypatching httpx.AsyncClient must not
    # recurse into itself when we construct the underlying transport client.
    _RealAsyncClient = httpx.AsyncClient

    class _Client:
        def __init__(self, *args, **kwargs):
            self._c = _RealAsyncClient(transport=httpx.MockTransport(handler))

        async def __aenter__(self):
            return self._c

        async def __aexit__(self, *exc):
            await self._c.aclose()

        def post(self, *args, **kwargs):
            return self._c.post(*args, **kwargs)

    return _Client


async def test_transcribe_openai_success(monkeypatch) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        assert request.headers.get("authorization") == "Bearer sk-test"
        return httpx.Response(200, json={"text": "hello world"})

    monkeypatch.setattr(tool_audio.httpx, "AsyncClient", _mock_openai_client(handler))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    text = await transcribe_bytes(b"dummy-bytes", "audio/mpeg")
    assert text == "hello world"
    assert "audio/transcriptions" in captured["url"]


async def test_transcribe_no_backend_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(tool_audio.TranscribeUnavailable):
        await transcribe_bytes(b"x", "audio/mpeg")


async def test_audio_tool_friendly_error_without_backend(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plugin = AudioToolPlugin()
    (spec,) = plugin.get_tool_specs(SessionContext())
    result = await plugin.execute_tool(
        context=SessionContext(),
        tool=spec,
        arguments={"data": base64.b64encode(b"x").decode(), "mime": "audio/mpeg"},
    )
    assert result is not None
    assert result.status.value == "error"
    assert "无可用转写后端" in (result.error or "")


async def test_audio_tool_ok_with_mocked_backend(monkeypatch) -> None:
    async def fake_transcribe(data: bytes, mime: str) -> str:
        return "transcribed text"

    monkeypatch.setattr(tool_audio, "transcribe_bytes", fake_transcribe)
    plugin = AudioToolPlugin()
    (spec,) = plugin.get_tool_specs(SessionContext())
    result = await plugin.execute_tool(
        context=SessionContext(),
        tool=spec,
        arguments={"data": base64.b64encode(b"x").decode(), "mime": "audio/wav"},
    )
    assert result is not None
    assert result.status.value == "ok"
    assert result.output["text"] == "transcribed text"


async def test_audio_tool_requires_source_or_data(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plugin = AudioToolPlugin()
    (spec,) = plugin.get_tool_specs(SessionContext())
    result = await plugin.execute_tool(
        context=SessionContext(), tool=spec, arguments={"mime": "audio/mpeg"}
    )
    assert result is not None
    assert result.status.value == "error"
    assert "source" in (result.error or "") or "data" in (result.error or "")
