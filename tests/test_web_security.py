"""Security & correctness regression tests for web_ui and tool_web.

Pins down the fixes from the code review:

* WebSocket handshake authorization (anti-CSWSH): cross-origin rejected,
  optional shared token enforced, allowlist configurable via env.
* Stream events are **unicast** to the requesting connection (no cross-user
  leakage) and carry ``session_id``.
* ``broadcast`` sends concurrently and prunes dead clients (backpressure).
* ``web_fetch`` SSRF guard: private/loopback/metadata addresses and non-http
  schemes are rejected before any request is made.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pyharness.plugins.tool_web import WebPlugin, _assert_public_http_url
from pyharness.plugins.web_ui import WebUIPlugin
from pyharness.schema import ToolResultStatus


# --------------------------------------------------------------------------- #
# WebSocket authorization (#4)
# --------------------------------------------------------------------------- #
def _plugin(port: int = 3080, token: str | None = None) -> WebUIPlugin:
    import os

    if token:
        os.environ["PYHARNESS_WEB_TOKEN"] = token
    else:
        os.environ.pop("PYHARNESS_WEB_TOKEN", None)
    try:
        return WebUIPlugin(port=port)
    finally:
        os.environ.pop("PYHARNESS_WEB_TOKEN", None)


def test_ws_cross_origin_rejected() -> None:
    p = _plugin()
    assert p._ws_authorized("http://evil.example.com", None) is False


def test_ws_local_origin_allowed() -> None:
    p = _plugin()
    assert p._ws_authorized("http://127.0.0.1:3080", None) is True
    assert p._ws_authorized("http://localhost:3080", None) is True


def test_ws_token_required_when_configured() -> None:
    p = _plugin(token="s3cret")
    # no origin (local tooling) but wrong/missing token => rejected
    assert p._ws_authorized(None, None) is False
    assert p._ws_authorized(None, "wrong") is False
    assert p._ws_authorized("http://127.0.0.1:3080", "s3cret") is True


def test_ws_env_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYHARNESS_ALLOWED_ORIGINS", "https://ui.example.com")
    p = WebUIPlugin(port=3080)
    assert p._ws_authorized("https://ui.example.com", None) is True
    assert p._ws_authorized("http://127.0.0.1:3080", None) is False


# --------------------------------------------------------------------------- #
# Unicast streaming + backpressure broadcast (#6 / #7)
# --------------------------------------------------------------------------- #
class _FakeWS:
    """Minimal WebSocket double: records sent frames, can simulate failure."""

    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.sent: list[str] = []

    async def send_text(self, message: str) -> None:
        if self.fail:
            raise RuntimeError("connection closed")
        self.sent.append(message)


@pytest.mark.asyncio
async def test_unicast_reaches_only_target() -> None:
    p = _plugin()
    a, b = _FakeWS("a"), _FakeWS("b")
    p.ws_clients.update({a, b})

    await p._send(a, "llm_stream_chunk", {"session_id": "s1", "content": "hi"})

    assert len(a.sent) == 1
    assert b.sent == []  # the other user must NOT receive it
    payload = json.loads(a.sent[0])
    assert payload["data"]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_unicast_to_none_is_noop() -> None:
    p = _plugin()
    await p._send(None, "error", {"message": "x"})  # must not raise


@pytest.mark.asyncio
async def test_broadcast_prunes_dead_clients_and_delivers() -> None:
    p = _plugin()
    slow, dead = _FakeWS("slow"), _FakeWS("dead")
    p.ws_clients.update({slow, dead})
    import asyncio

    started: list[str] = []

    async def _slow_send(msg: str) -> None:
        started.append("slow")
        await asyncio.sleep(0.05)
        await _FakeWS.send_text(slow, msg)

    async def _dead_send(msg: str) -> None:
        raise RuntimeError("closed")

    slow.send_text = _slow_send  # type: ignore[method-assign]
    dead.send_text = _dead_send  # type: ignore[method-assign]

    await p.broadcast("system", {"hello": True})

    assert slow in p.ws_clients and slow.sent  # delivered despite slowness
    assert dead not in p.ws_clients  # pruned after failure


# --------------------------------------------------------------------------- #
# SSRF guard (#5)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:3080/",            # loopback (our own UI!)
        "http://localhost/admin",            # loopback by name
        "http://10.0.0.5/internal",          # private
        "http://172.16.1.1/",                # private
        "http://192.168.1.10/router",        # private
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "file:///etc/passwd",                # wrong scheme
        "ftp://example.com/x",               # wrong scheme
    ],
)
def test_ssrf_blocked(url: str) -> None:
    with pytest.raises(PermissionError):
        _assert_public_http_url(url)


def test_ssrf_allows_public_https() -> None:
    _assert_public_http_url("https://example.com/page")  # must not raise


@pytest.mark.asyncio
async def test_fetch_private_url_returns_error() -> None:
    """End-to-end through the tool: private URL is refused without a request."""
    plugin = WebPlugin()
    if not plugin._available:  # pragma: no cover
        pytest.skip("web extras not installed")
    result = await plugin._fetch({"url": "http://127.0.0.1:3080/api/sessions"})
    assert result.status == ToolResultStatus.ERROR
    assert "已拦截" in (result.error or "")