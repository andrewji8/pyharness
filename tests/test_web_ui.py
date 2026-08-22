"""Tests for the Web UI plugin (FastAPI + WebSocket)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from pyharness import Harness
from pyharness.plugins.llm import entry as llm
from pyharness.plugins.web_ui import WebUIPlugin
from pyharness.schema import LLMResponse


def _setup_harness():
    from pyharness.schema import HarnessConfig
    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    llm.clear()
    llm.use_dummy(models=("demo",), plan=[LLMResponse(model="demo", content="ok")])
    h.register_plugin(llm)
    web_ui = WebUIPlugin()
    h.register_plugin(web_ui)
    h.initialize()
    return h, web_ui


class TestWebUIPluginRest:
    def test_chat_returns_session_id(self):
        h, web_ui = _setup_harness()
        client = TestClient(web_ui.app)
        resp = client.post("/api/chat", json={"content": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["status"] == "ok"

    def test_list_sessions_returns_empty_initially(self):
        h, web_ui = _setup_harness()
        client = TestClient(web_ui.app)
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data

    def test_get_session_not_found(self):
        h, web_ui = _setup_harness()
        client = TestClient(web_ui.app)
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_get_plan_not_found(self):
        h, web_ui = _setup_harness()
        client = TestClient(web_ui.app)
        resp = client.get("/api/plans/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_search_returns_empty(self):
        h, web_ui = _setup_harness()
        client = TestClient(web_ui.app)
        resp = client.post("/api/search", json={"query": "test", "session_id": "nonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_index_returns_html(self):
        h, web_ui = _setup_harness()
        client = TestClient(web_ui.app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestWebUIPluginWebSocket:
    def test_websocket_connection(self):
        h, web_ui = _setup_harness()
        client = TestClient(web_ui.app)
        with client.websocket_connect("/ws/events") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            # Should not error

    def test_websocket_broadcast(self):
        h, web_ui = _setup_harness()
        client = TestClient(web_ui.app)
        with client.websocket_connect("/ws/events") as ws:
            ws.send_text(json.dumps({"type": "user_message", "content": "hello"}))
            # Should receive events without error


__all__ = ["TestWebUIPluginRest", "TestWebUIPluginWebSocket"]
