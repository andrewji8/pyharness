"""Multimodal Batch 1 regression tests.

Verifies the contract described in the batch spec:

1. Legacy messages without ``parts`` round-trip through SQLite as ``parts == ()``.
2. ``image`` / ``text`` parts map to the OpenAI content-array structure; an empty
   ``parts`` keeps the plain string ``content`` path untouched.
3. ``parts`` survive a SQLite save/load round-trip without loss.
4. Vision routing switches the effective model when an image part is present and
   the current model is outside the vision whitelist.
"""

from __future__ import annotations

import json

import httpx
from typing import Any

from pyharness.context import SessionContext
from pyharness.plugins.llm.http import HTTPProvider
from pyharness.plugins.session_store import SQLiteSessionStorePlugin
from pyharness.schema import ContentPart, LLMRequest, Message, Role


def _provider(transport: httpx.MockTransport, model: str = "m") -> HTTPProvider:
    return HTTPProvider(models=(model,), base_url="http://test", transport=transport)


# --------------------------------------------------------------------------- #
# 1. Legacy (no parts) read/write compatibility
# --------------------------------------------------------------------------- #
async def test_legacy_message_without_parts_round_trips_empty(tmp_path) -> None:
    store = SQLiteSessionStorePlugin(db_path=str(tmp_path / "legacy.db"))
    await store.initialize()
    try:
        msg = Message(role=Role.USER, content="hello legacy")
        ctx = SessionContext(messages=(msg,))
        await store.save_session(ctx)

        data = await store.load_session(str(ctx.session_id))
        assert data is not None
        assert len(data.messages) == 1
        loaded = data.messages[0]
        assert loaded.content == "hello legacy"
        # Old rows (NULL parts) deserialize to an empty tuple — no crash.
        assert loaded.parts == ()
    finally:
        await store.teardown()


# --------------------------------------------------------------------------- #
# 2. parts -> OpenAI content array; empty parts -> plain string
# --------------------------------------------------------------------------- #
async def test_empty_parts_keeps_string_content() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(httpx.MockTransport(handler))
    await provider.chat(LLMRequest(model="m", messages=(Message(role=Role.USER, content="plain"),)))
    assert captured["body"]["messages"][0]["content"] == "plain"


async def test_image_part_maps_to_content_array() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(httpx.MockTransport(handler))
    msg = Message(
        role=Role.USER,
        content="",  # image-only message: no leading text block
        parts=(
            ContentPart(type="text", text="describe this"),
            ContentPart(type="image", url="http://example.com/pic.png"),
        ),
    )
    await provider.chat(LLMRequest(model="m", messages=(msg,)))

    content = captured["body"]["messages"][0]["content"]
    assert isinstance(content, list)
    types = [b["type"] for b in content]
    assert "text" in types
    assert "image_url" in types
    image_block = next(b for b in content if b["type"] == "image_url")
    assert image_block["image_url"]["url"] == "http://example.com/pic.png"


async def test_content_string_prepended_as_first_text_part() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(httpx.MockTransport(handler))
    msg = Message(
        role=Role.USER,
        content="leading text",
        parts=(ContentPart(type="text", text="describe this"),),
    )
    await provider.chat(LLMRequest(model="m", messages=(msg,)))

    blocks = captured["body"]["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "leading text"}
    assert blocks[1] == {"type": "text", "text": "describe this"}


async def test_base64_image_part_uses_data_url() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(httpx.MockTransport(handler))
    msg = Message(
        role=Role.USER,
        content="",
        parts=(ContentPart(type="image", url="data:image/png;base64,aGVsbG8="),),
    )
    await provider.chat(LLMRequest(model="m", messages=(msg,)))

    image_block = next(
        b for b in captured["body"]["messages"][0]["content"] if b["type"] == "image_url"
    )
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


# --------------------------------------------------------------------------- #
# 3. parts persistence round-trip
# --------------------------------------------------------------------------- #
async def test_parts_persist_round_trip(tmp_path) -> None:
    store = SQLiteSessionStorePlugin(db_path=str(tmp_path / "parts.db"))
    await store.initialize()
    try:
        msg = Message(
            role=Role.USER,
            content="see",
            parts=(
                ContentPart(type="text", text="caption"),
                ContentPart(type="image", url="http://x/y.png"),
            ),
        )
        ctx = SessionContext(messages=(msg,))
        await store.save_session(ctx)

        data = await store.load_session(str(ctx.session_id))
        assert data is not None
        loaded = data.messages[0]
        assert len(loaded.parts) == 2
        assert loaded.parts[0].type == "text"
        assert loaded.parts[0].text == "caption"
        assert loaded.parts[1].type == "image"
        assert loaded.parts[1].url == "http://x/y.png"
    finally:
        await store.teardown()


# --------------------------------------------------------------------------- #
# 4. Vision routing
# --------------------------------------------------------------------------- #
async def test_vision_routing_switches_model(monkeypatch) -> None:
    import pyharness.plugins.llm.http as http_mod

    monkeypatch.setattr(http_mod, "_VISION_MODEL", "vision-model")
    # Whitelist = models that already understand images; "base-model" is NOT in
    # it, so an image request must be transparently switched to vision-model.
    monkeypatch.setattr(http_mod, "_VISION_WHITELIST", {"vision-capable"})

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(httpx.MockTransport(handler), model="base-model")
    msg = Message(
        role=Role.USER,
        content="pic",
        parts=(ContentPart(type="image", url="http://x/y.png"),),
    )
    req = LLMRequest(model="base-model", messages=(msg,))

    # The routing helper itself must switch the model for this request.
    assert provider._effective_model(req) == "vision-model"
    await provider.chat(req)
    assert captured["model"] == "vision-model"


async def test_vision_routing_skipped_when_no_image(monkeypatch) -> None:
    import pyharness.plugins.llm.http as http_mod

    monkeypatch.setattr(http_mod, "_VISION_MODEL", "vision-model")
    monkeypatch.setattr(http_mod, "_VISION_WHITELIST", {"base-model"})

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(httpx.MockTransport(handler), model="base-model")
    req = LLMRequest(model="base-model", messages=(Message(role=Role.USER, content="hi"),))

    assert provider._effective_model(req) == "base-model"
    await provider.chat(req)
    assert captured["model"] == "base-model"


async def test_vision_routing_with_only_vision_model_env(monkeypatch) -> None:
    """Fix A: only PYHARNESS_VISION_MODEL set (empty whitelist) still switches."""
    import pyharness.plugins.llm.http as http_mod

    monkeypatch.setattr(http_mod, "_VISION_MODEL", "vision-model")
    monkeypatch.setattr(http_mod, "_VISION_WHITELIST", set())  # empty whitelist

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = _provider(httpx.MockTransport(handler), model="base-model")
    msg = Message(
        role=Role.USER,
        content="pic",
        parts=(ContentPart(type="image", url="http://x/y.png"),),
    )
    req = LLMRequest(model="base-model", messages=(msg,))

    assert provider._effective_model(req) == "vision-model"
    await provider.chat(req)
    assert captured["model"] == "vision-model"
