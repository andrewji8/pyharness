"""Redis Pub/Sub distributed event bus plugin (cross-instance).

In single-process mode PyHarness fans out events through an in-process pluggy
``observe`` hook. When multiple ``pyharness serve`` instances sit behind a load
balancer, events emitted by instance A (e.g. ``tool.stream``, LLM stream
chunks) would never reach instance B's WebSocket clients.

This plugin bridges instances over Redis Pub/Sub:

* On ``observe`` it serialises the event (``type``/``session_id``/JSON-safe
  ``payload``, plus ``source_instance_id``) and publishes it to a global channel.
* A background :class:`asyncio.Task` subscribes to that channel. Incoming
  messages whose ``source_instance_id`` matches this process are dropped
  (anti-storm / no infinite re-broadcast). Everything else is re-injected into
  the *local* ``observe`` hook so local observers (WebSocket clients, CLI)
  deliver the remote event.

Graceful degradation: without ``REDIS_URL`` (or without the ``redis`` package)
this plugin does nothing, leaving the existing in-process event bus untouched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from pluggy import HookimplMarker

from pyharness.schema import Event

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")

# Global Redis channel used to fan out events across instances.
DEFAULT_CHANNEL = "pyharness:events"


def _jsonable(value: Any) -> Any:
    """Recursively convert an object graph to JSON-safe primitives."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        import base64

        return {"__kind__": "bytes", "data": base64.b64encode(value).decode("ascii")}
    # pydantic v2
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict") and callable(value.dict):  # pydantic v1 fallback
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    if isinstance(value, uuid.UUID):
        return {"__kind__": "uuid", "value": str(value)}
    if isinstance(value, datetime):
        return {"__kind__": "datetime", "value": value.isoformat()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, set):
        return {"__kind__": "set", "value": [_jsonable(v) for v in value]}
    # enums
    if hasattr(value, "value"):
        return _jsonable(value.value)
    # Fall back to repr for unknown objects (should not normally happen).
    return str(value)


def _from_jsonable(value: Any) -> Any:
    """Reverse of :func:`_jsonable` — reconstruct plain data structures."""
    if isinstance(value, dict):
        kind = value.get("__kind__")
        if kind == "bytes":
            import base64

            return base64.b64decode(value["data"])
        if kind == "uuid":
            return uuid.UUID(value["value"])
        if kind == "datetime":
            return datetime.fromisoformat(value["value"])
        if kind == "set":
            return {_from_jsonable(v) for v in value["value"]}
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    return value


class RedisEventBusPlugin:
    """Bridge local engine events to a distributed Redis Pub/Sub channel.

    Parameters
    ----------
    redis_url:
        Redis connection URL. If ``None``, read from ``REDIS_URL`` env var.
    channel:
        Redis Pub/Sub channel name. Defaults to ``pyharness:events``.
    """

    def __init__(self, redis_url: str | None = None, channel: str = DEFAULT_CHANNEL) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.channel = channel
        # Unique per-process id used to drop our own broadcast (anti-storm).
        self.instance_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._harness: Any = None
        self._redis: Any = None
        self._pubsub: Any = None
        self._subscriber: asyncio.Task[Any] | None = None
        self._applying_remote = False
        self._started = False

    # -- lifecycle ---------------------------------------------------------- #
    @hookimpl
    def harness_initialized(self, harness: Any) -> None:
        """Capture the harness and start the background subscriber."""
        self._harness = harness
        if not self.redis_url:
            logger.info("RedisEventBusPlugin: no REDIS_URL, staying in-process")
            return
        try:
            import redis.asyncio as redis  # noqa: F401
        except ImportError:
            logger.warning("redis library not found; RedisEventBusPlugin disabled")
            return
        self._start()
        self._started = True

    @hookimpl
    def harness_shutdown(self, harness: Any) -> None:
        """Cancel the subscriber and close connections on shutdown."""
        if self._subscriber is not None:
            self._subscriber.cancel()
            self._subscriber = None
        if self._pubsub is not None:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._pubsub.aclose())
                else:
                    asyncio.run(self._pubsub.aclose())
            except Exception:
                logger.warning("failed to close redis pubsub", exc_info=True)
            self._pubsub = None
        self._redis = None

    def _start(self) -> None:
        """Spawn the background Redis subscribe task (idempotent)."""
        if self._subscriber is not None and not self._subscriber.done():
            return
        self._subscriber = asyncio.create_task(self._run_subscriber())

    async def _run_subscriber(self) -> None:
        """Continuously read from the Redis channel and inject events locally."""
        import redis.asyncio as redis

        try:
            if self._redis is None:
                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=10,
                )
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(self.channel)
            logger.info(
                "RedisEventBusPlugin subscribed to %s (instance=%s)",
                self.channel,
                self.instance_id,
            )
            async for message in self._pubsub.listen():
                if message.get("type") != "message":
                    continue
                self._handle_remote(message.get("data"))
        except asyncio.CancelledError:
            logger.info("RedisEventBusPlugin subscriber cancelled")
        except Exception:
            logger.warning("RedisEventBusPlugin subscriber error", exc_info=True)

    def _handle_remote(self, data: Any) -> None:
        """Deserialise a remote event and fan it out to local observers."""
        try:
            envelope = json.loads(data) if isinstance(data, str) else data
        except (TypeError, ValueError):
            logger.warning("RedisEventBusPlugin: dropped malformed message")
            return

        if not isinstance(envelope, dict):
            return
        # Anti-storm: never re-inject our own broadcast.
        if envelope.get("source_instance_id") == self.instance_id:
            return

        try:
            event = Event(
                type=envelope["type"],
                session_id=uuid.UUID(envelope["session_id"]),
                payload=_from_jsonable(envelope.get("payload", {})),
                timestamp=datetime.fromisoformat(envelope["timestamp"]),
            )
        except (KeyError, ValueError) as exc:
            logger.warning("RedisEventBusPlugin: dropped invalid event: %s", exc)
            return

        # Re-inject into the local observe chain with the reentrancy guard set,
        # so we do not re-publish a remote event back onto the channel.
        ctx = self._make_remote_context(event)
        if self._harness is not None:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._dispatch_local(ctx, event))

    async def _dispatch_local(self, ctx: Any, event: Event) -> None:
        # Guard must be held for the lifetime of the local fan-out (not just
        # while scheduling), or observe() would re-publish a remote event.
        try:
            self._applying_remote = True
            await self._harness.bus.pm.hook.observe(context=ctx, event=event)
        except Exception:
            logger.warning("RedisEventBusPlugin: local observe failed", exc_info=True)
        finally:
            self._applying_remote = False

    def _make_remote_context(self, event: Event) -> Any:
        """Build a minimal SessionContext so local observers can process a
        remote event without needing the full in-process session state."""
        from pyharness.context import SessionContext

        return SessionContext(
            session_id=event.session_id,
            namespace="default",
            messages=(),
        )

    # -- observer hook ------------------------------------------------------ #
    @hookimpl
    async def observe(self, context: Any, event: Any) -> None:
        """Publish a locally-emitted event to the Redis channel.

        Reentrancy-safe: while applying a *remote* event, publication is
        skipped so we never forward someone else's event back onto the bus.
        """
        if not self._started or self._redis is None:
            return
        if self._applying_remote:
            return
        # Events with a leading "_" are internal (WebSocketObserver skips them
        # too); don't broadcast those across instances.
        if event.type.startswith("_"):
            return

        envelope = {
            "source_instance_id": self.instance_id,
            "type": event.type,
            "session_id": str(event.session_id),
            "timestamp": event.timestamp.isoformat(),
            "payload": _jsonable(event.payload),
        }
        try:
            await self._redis.publish(self.channel, json.dumps(envelope))
        except Exception:
            logger.warning("RedisEventBusPlugin: publish failed", exc_info=True)


__all__ = ["RedisEventBusPlugin", "DEFAULT_CHANNEL"]
