"""Tests for the Redis Pub/Sub distributed event bus plugin.

Uses fakeredis (shared FakeServer across "instances") to verify cross-instance
broadcast and the anti-storm ``source_instance_id`` filter without a real Redis.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

import pytest

for _dep in ("redis", "fakeredis.aioredis"):
    try:
        __import__(_dep)
    except ImportError:
        pytest.skip(f"{_dep} not installed; skipping Redis event bus tests", allow_module_level=True)

from pluggy import HookimplMarker

from pyharness import Harness
from pyharness.context import SessionContext
from pyharness.plugins.event_bus_redis import DEFAULT_CHANNEL, RedisEventBusPlugin, _from_jsonable, _jsonable
from pyharness.plugins.llm import entry as llm
from pyharness.schema import (
    ContentPart,
    Event,
    HarnessConfig,
    Role,
    ToolStreamEvent,
)


def _shared_redis_pairs():
    """Return two fakeredis.aioredis clients sharing the same FakeServer.

    Sharing the server lets pub/sub messages flow between the two "instances".
    """
    import fakeredis.aioredis
    import fakeredis

    server = fakeredis.FakeServer()
    a = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    b = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return a, b


class _RecordingObserver:
    """A pluggy plugin whose ``observe`` records delivered events."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    @HookimplMarker("pyharness")
    def observe(self, context: SessionContext, event: Event) -> None:
        self.events.append(event)


def _harness() -> Harness:
    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    return h


def _make_event(event_type: str = "tool.stream", payload: dict | None = None) -> Event:
    return Event(
        type=event_type,
        session_id=uuid.uuid4(),
        payload=payload or {},
        timestamp=datetime.now(timezone.utc),
    )


async def _wait_for(predicate, timeout: float = 3.0) -> bool:
    """Poll until ``predicate`` returns truthy or timeout expires."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


def test_jsonable_preserves_complex_payload() -> None:
    """Serialization round-trips complex payloads (pydantic, uuid, datetime)."""
    payload = {
        "tool_stream_event": ToolStreamEvent(
            tool_name="python_exec",
            stream_type="stdout",
            content="hello\n",
        ),
        "parts": (ContentPart(type="text", text="hi"),),
        "session_ref": uuid.uuid4(),
        "when": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "count": 3,
        "flag": True,
        "data": b"\x00\x01",
    }
    encoded = _jsonable(payload)
    # Must be JSON-serializable.
    text = json.dumps(encoded)
    decoded = _from_jsonable(json.loads(text))

    assert decoded["count"] == 3
    assert decoded["flag"] is True
    assert decoded["data"] == b"\x00\x01"
    assert decoded["when"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    # pydantic model dumped to its JSON dict representation.
    assert decoded["tool_stream_event"]["tool_name"] == "python_exec"
    assert decoded["tool_stream_event"]["stream_type"] == "stdout"
    assert decoded["tool_stream_event"]["content"] == "hello\n"
    assert decoded["parts"][0]["type"] == "text"
    assert str(decoded["session_ref"]) == str(payload["session_ref"])


async def test_cross_instance_broadcast_via_pubsub() -> None:
    """Instance A publishes; instance B's subscriber delivers it locally."""
    ra, rb = _shared_redis_pairs()

    # Instance B: harness + recording observer + event bus plugin (subscriber).
    hb = _harness()
    rec_b = _RecordingObserver()
    hb.register_plugin(rec_b)
    eb_b = RedisEventBusPlugin(redis_url="redis://fake", channel=DEFAULT_CHANNEL)
    eb_b._harness = hb
    eb_b._redis = rb
    eb_b._start()
    eb_b._started = True
    try:
        # Give B's subscriber time to establish the subscription.
        ok = await _wait_for(lambda: eb_b._pubsub is not None)
        assert ok, "instance B subscriber did not start"

        # Instance A: publish a tool.stream event.
        ra_pub = RedisEventBusPlugin(redis_url="redis://fake", channel=DEFAULT_CHANNEL)
        ra_pub._redis = ra
        ra_pub._started = True
        event = _make_event(
            "tool.stream",
            {"tool_stream_event": ToolStreamEvent(tool_name="py", stream_type="log", content="run")},
        )
        ctx = SessionContext(session_id=event.session_id)
        await ra_pub.observe(ctx, event)

        # B should have received and re-injected the event locally.
        ok = await _wait_for(lambda: len(rec_b.events) > 0, timeout=3.0)
        assert ok, "instance B never received instance A's event"
        delivered = rec_b.events[0]
        assert delivered.type == "tool.stream"
        assert str(delivered.session_id) == str(event.session_id)
        assert delivered.payload["tool_stream_event"]["content"] == "run"
    finally:
        eb_b._subscriber.cancel()
        await asyncio.sleep(0.05)


async def test_anti_storm_drops_own_broadcast() -> None:
    """A publisher must not re-inject its own echoed broadcast."""
    ra, rb = _shared_redis_pairs()

    hb = _harness()
    rec_b = _RecordingObserver()
    hb.register_plugin(rec_b)

    eb = RedisEventBusPlugin(redis_url="redis://fake", channel=DEFAULT_CHANNEL)
    eb._harness = hb
    eb._redis = rb
    eb._start()
    eb._started = True
    try:
        await _wait_for(lambda: eb._pubsub is not None)

        event = _make_event("tool.result", {"ok": True})

        # Simulate receiving our own broadcast: same source_instance_id.
        envelope = {
            "source_instance_id": eb.instance_id,
            "type": event.type,
            "session_id": str(event.session_id),
            "timestamp": event.timestamp.isoformat(),
            "payload": _jsonable(event.payload),
        }
        eb._handle_remote(json.dumps(envelope))
        # Give any dispatch task a chance to run (it must NOT run).
        await asyncio.sleep(0.15)
        assert rec_b.events == [], "own broadcast must be dropped (anti-storm)"

        # Sanity: a different instance's event IS delivered.
        other = RedisEventBusPlugin(redis_url="redis://fake")
        envelope2 = {
            "source_instance_id": other.instance_id,
            "type": "tool.stream",
            "session_id": str(event.session_id),
            "timestamp": event.timestamp.isoformat(),
            "payload": {"text": "from-other-instance"},
        }
        eb._handle_remote(json.dumps(envelope2))
        ok = await _wait_for(lambda: len(rec_b.events) > 0)
        assert ok, "foreign event was not delivered"
        assert rec_b.events[0].payload["text"] == "from-other-instance"
    finally:
        eb._subscriber.cancel()


async def test_remote_event_not_returned_to_channel(monkeypatch) -> None:
    """While applying a remote event, observe() must not re-publish it."""
    ra, rb = _shared_redis_pairs()

    hb = _harness()
    rec_b = _RecordingObserver()
    hb.register_plugin(rec_b)

    eb = RedisEventBusPlugin(redis_url="redis://fake", channel=DEFAULT_CHANNEL)
    eb._harness = hb
    eb._redis = rb
    eb._start()
    eb._started = True
    published: list[dict] = []

    async def fake_publish(channel, message):
        published.append(json.loads(message))

    monkeypatch.setattr(eb._redis, "publish", fake_publish)
    try:
        await _wait_for(lambda: eb._pubsub is not None)

        other = RedisEventBusPlugin(redis_url="redis://fake")
        event = _make_event("tool.stream", {"text": "remote-payload"})
        envelope = {
            "source_instance_id": other.instance_id,
            "type": event.type,
            "session_id": str(event.session_id),
            "timestamp": event.timestamp.isoformat(),
            "payload": _jsonable(event.payload),
        }
        eb._handle_remote(json.dumps(envelope))
        await _wait_for(lambda: len(rec_b.events) > 0)
        await asyncio.sleep(0.15)
        assert published == [], "remote event must not be re-published"
    finally:
        eb._subscriber.cancel()


async def test_factory_registers_event_bus_when_redisd_set(monkeypatch) -> None:
    """Factory should register RedisEventBusPlugin when REDIS_URL is set."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    ra, _rb = _shared_redis_pairs()

    async def mock_initialize(self) -> None:
        self._redis = ra
        self._initialized = True

    monkeypatch.setattr("pyharness.plugins.session_store_redis.RedisSessionStorePlugin.initialize", mock_initialize)

    import importlib
    from pyharness import factory

    importlib.reload(factory)

    try:
        harness = factory.build_harness(model="dummy", provider="dummy")
        from pyharness.plugins.event_bus_redis import RedisEventBusPlugin

        plugins = list(harness.bus.pm.get_plugins())
        assert any(isinstance(p, RedisEventBusPlugin) for p in plugins), (
            "RedisEventBusPlugin not registered when REDIS_URL set"
        )
    finally:
        importlib.reload(factory)
        monkeypatch.delenv("REDIS_URL", raising=False)
