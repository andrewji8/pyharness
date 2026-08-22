"""Tests for module C — the Web/API plugin.

Uses FastAPI's ``TestClient`` (in-process, no sockets) against the ``dummy``
provider: REST one-shot, REST multi-turn continuation by ``session_id``, and
WebSocket streaming + reset.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pyharness.plugins.llm import entry as llm
from pyharness.plugins.web.app import app, store
from pyharness.schema import LLMResponse

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Isolate state between tests (global llm registry + in-memory store)."""
    llm.clear()
    store.clear()
    yield
    store.clear()


def _plan(*contents: str, chunk_size: int = 2) -> None:
    llm.use_dummy(
        models=("demo",),
        plan=[LLMResponse(model="demo", content=c) for c in contents],
        chunk_size=chunk_size,
    )


def _drain_until_end(ws) -> dict:
    while True:
        frame = ws.receive_json()
        if frame["type"] == "end":
            return frame


# --------------------------------------------------------------------------- #
def test_health_reports_configured_models() -> None:
    _plan("hi")
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "demo" in data["models"]


def test_chat_single_turn() -> None:
    _plan("hello from api")
    r = client.post("/chat", json={"text": "ping", "model": "demo"})
    assert r.status_code == 200
    data = r.json()
    assert data["reply"] == "hello from api"
    assert data["session_id"]
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]


def test_chat_continues_by_session_id() -> None:
    _plan("first", "second")
    r1 = client.post("/chat", json={"text": "Q1", "model": "demo"}).json()
    r2 = client.post(
        "/chat", json={"text": "Q2", "model": "demo", "session_id": r1["session_id"]}
    ).json()
    assert r2["reply"] == "second"
    assert [m["content"] for m in r2["messages"]] == ["Q1", "first", "Q2", "second"]
    assert r2["session_id"] == r1["session_id"]  # same reversible lineage


# --------------------------------------------------------------------------- #
def test_ws_streams_deltas_then_ends() -> None:
    _plan("hello world", chunk_size=3)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"text": "hi", "model": "demo"})
        chunks: list[str] = []
        frame = ws.receive_json()
        while frame["type"] != "end":
            chunks.append(frame["delta"])
            frame = ws.receive_json()
        assert "".join(chunks) == "hello world"
        assert frame["reply"] == "hello world"
        assert frame["session_id"]


def test_ws_continues_then_resets_lineage() -> None:
    _plan("reply-1", "reply-2", chunk_size=10)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"text": "A", "model": "demo"})
        end1 = _drain_until_end(ws)
        assert end1["reply"] == "reply-1"

        # second turn continues the same session
        ws.send_json({"text": "B", "model": "demo"})
        end2 = _drain_until_end(ws)
        assert end2["reply"] == "reply-2"
        assert end2["session_id"] == end1["session_id"]

        # reset starts a fresh lineage
        ws.send_json({"action": "reset"})
        assert ws.receive_json()["type"] == "reset_ok"
        ws.send_json({"text": "C", "model": "demo"})
        end3 = _drain_until_end(ws)
        assert end3["session_id"] != end2["session_id"]