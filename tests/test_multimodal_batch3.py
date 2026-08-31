"""Batch 3/3 tests: WebSocket multimodal intake, audio transcription, CLI --image."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from pyharness.context import SessionContext
from pyharness.plugins.web_ui import WebUIPlugin, _prepare_user_message
from pyharness.plugins.tool_audio import transcribe_bytes
from pyharness.schema import LLMStreamChunk
from pyharness.plugins.session_store import SQLiteSessionStorePlugin


# ---------------------------------------------------------------------------
# WebSocket fixtures / helpers
# ---------------------------------------------------------------------------
class _WsFakeHarness:
    """Minimal harness stub that records the initial_message it was given."""

    def __init__(self) -> None:
        self.captured = None
        self.last_context = None

    async def stream_session(
        self,
        agent,
        initial_text=None,
        *,
        initial_message=None,
        continue_from=None,
        resume_session_id=None,
        namespace=None,
    ):
        self.captured = initial_message
        self.last_context = SessionContext()
        yield LLMStreamChunk(delta="ok", tool_calls=[])


async def _dummy_model(model=None):
    return "dummy"


def _make_plugin():
    plugin = WebUIPlugin()
    plugin.harness = _WsFakeHarness()
    plugin._get_model = _dummy_model
    return plugin


# ---------------------------------------------------------------------------
# 1. WebSocket receives image/audio parts -> Message built & stored
# ---------------------------------------------------------------------------
async def test_ws_message_builds_parts_and_stores(monkeypatch) -> None:
    # No transcription backend: audio part kept, transcription simply skipped.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    msg = await _prepare_user_message(
        "hi",
        [
            {"type": "image", "url": "data:image/png;base64,AAAA"},
            {"type": "audio", "url": "data:audio/wav;base64,BBBB"},
        ],
    )
    assert msg.role.value == "user"
    types = [p.type for p in msg.parts]
    assert "image" in types
    assert "audio" in types

    # Round-trip through the real SQLite session store ("存入 DB").
    store = SQLiteSessionStorePlugin()
    await store.initialize()
    try:
        ctx = SessionContext(messages=(msg,))
        await store.save_session(ctx)
        data = await store.load_session(str(ctx.session_id))
        loaded = data.messages[0]
        assert any(p.type == "image" for p in loaded.parts)
        assert any(p.type == "audio" for p in loaded.parts)
    finally:
        await store.teardown()


# ---------------------------------------------------------------------------
# 2. WebSocket audio part -> transcription appends a text part (mocked)
# ---------------------------------------------------------------------------
async def test_ws_audio_transcription_appends_text_part(monkeypatch) -> None:
    async def fake_transcribe(data: bytes, mime: str) -> str:
        return "转录文本"

    import pyharness.plugins.web_ui as web_ui_mod

    monkeypatch.setattr(web_ui_mod, "transcribe_bytes", fake_transcribe)

    msg = await _prepare_user_message(
        "", [{"type": "audio", "url": "data:audio/wav;base64,BBBB"}]
    )
    text_parts = [p for p in msg.parts if p.type == "text"]
    assert text_parts, "expected a transcribed text part to be appended"
    assert any("转录文本" in (p.text or "") for p in text_parts)


# ---------------------------------------------------------------------------
# 3. CLI --image option parses files into image parts
# ---------------------------------------------------------------------------
def test_cli_build_image_parts(tmp_path) -> None:
    from pyharness.plugins.cli.app import _build_image_parts

    f = tmp_path / "a.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    parts = _build_image_parts([str(f)])
    assert len(parts) == 1
    assert parts[0].type == "image"
    assert parts[0].url.startswith("data:image/png;base64,")


def test_cli_build_image_parts_missing_raises() -> None:
    import pytest
    import typer

    from pyharness.plugins.cli.app import _build_image_parts

    with pytest.raises(typer.BadParameter):
        _build_image_parts(["/no/such/file.png"])


async def test_cli_stream_turn_sends_image_parts() -> None:
    from pyharness.plugins.cli.app import _stream_turn
    from pyharness.schema import AgentConfig, ContentPart

    class FakeHarness:
        def __init__(self) -> None:
            self.last = None
            self.last_context = None

        async def stream_session(
            self,
            agent,
            initial_text=None,
            *,
            initial_message=None,
            continue_from=None,
            namespace=None,
        ):
            self.last = initial_message
            self.last_context = SessionContext()
            yield LLMStreamChunk(delta="x", tool_calls=[])

    h = FakeHarness()
    agent = AgentConfig(name="a", model="dummy")
    parts = [ContentPart(type="image", url="data:image/png;base64,AAAA")]
    await _stream_turn(h, agent, "describe this", None, parts)
    assert h.last is not None
    assert h.last.parts[0].type == "image"


# ---------------------------------------------------------------------------
# Eval: skip_when no_vision_model
# ---------------------------------------------------------------------------
async def test_eval_skip_when_no_vision_model(monkeypatch, tmp_path) -> None:
    from pyharness.plugins.eval_runner import EvalRunner, EvalTask

    monkeypatch.delenv("PYHARNESS_VISION_MODEL", raising=False)
    suite = tmp_path / "s.yaml"
    suite.write_text("tasks: []\n", encoding="utf-8")

    class _NoopHarness:
        config = type("C", (), {"model": "dummy", "namespace": "default"})()

    task = EvalTask(id="m", category="multimodal", prompt="x", skip_when="no_vision_model")
    runner = EvalRunner(_NoopHarness(), suite_path=suite, model="dummy")
    result = await runner.run_task(task)
    assert result.skipped is True


# ---------------------------------------------------------------------------
# WebSocket: large payload (Base64 image) must parse without dropping connection
# ---------------------------------------------------------------------------
async def test_ws_large_message_parsed() -> None:
    plugin = _make_plugin()
    big = "A" * (5 * 1024 * 1024)  # ~5MB Base64 payload (post-encoding inflation)
    payload = {
        "type": "user_message",
        "content": "big image",
        "parts": [{"type": "image", "url": f"data:image/png;base64,{big}"}],
    }
    with TestClient(plugin.app) as client:
        with client.websocket_connect("/ws/events") as ws:
            ws.send_json(payload)
            for _ in range(100):
                if plugin.harness.captured is not None:
                    break
                time.sleep(0.05)
            assert plugin.harness.captured is not None
            assert any(p.type == "image" for p in plugin.harness.captured.parts)


async def test_ws_malformed_json_does_not_crash() -> None:
    plugin = _make_plugin()
    with TestClient(plugin.app) as client:
        with client.websocket_connect("/ws/events") as ws:
            # First frame is garbage JSON — must be absorbed, not disconnect.
            ws.send_text("{this is not valid json")
            # Connection stays alive; a valid frame afterwards is still processed.
            ws.send_json({"type": "user_message", "content": "ok", "parts": []})
            for _ in range(100):
                if plugin.harness.captured is not None:
                    break
                time.sleep(0.05)
            assert plugin.harness.captured is not None
            assert plugin.harness.captured.content == "ok"
