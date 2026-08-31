"""Distributed tool execution: Redis-backed task queue + result delivery.

When ``PYHARNESS_DISTRIBUTED_EXEC=1`` **and** ``REDIS_URL`` is present, the web
node offloads tool execution onto independent ``pyharness worker`` processes so
compute scales horizontally instead of blocking web nodes.

Protocol
--------
* Queue: ``LPUSH pyharness:tool_tasks`` / worker ``BRPOP``.
* Result: worker ``LPUSH pyharness:tool_result:<task_id>``, web node
  ``BRPOP``-waits (with timeout).

*Why a List, not Pub/Sub, for results?* A List is durable and has no
subscription-timing race: the web node can ``BRPOP(timeout)`` and gets a result
even if the worker publishes before the web node's client was ready. Pub/Sub is
fire-and-forget — a result could be dropped if nobody is subscribed at that
instant, which would make the web node wait and then time out. Lists win on the
"worker must never leave the web node hanging" requirement. (Streaming
``tool.stream`` events still flow back over the batch-2 Redis Pub/Sub event
bus — see :mod:`pyharness.plugins.event_bus_redis`.)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any

from pyharness.schema import ToolResult, ToolResultStatus, ToolTask, ToolTaskResult

logger = logging.getLogger(__name__)

TASK_QUEUE_KEY = "pyharness:tool_tasks"
_RESULT_PREFIX = "pyharness:tool_result:"


def distributed_exec_enabled() -> bool:
    """Return True when distributed tool execution is switched on.

    Requires both ``PYHARNESS_DISTRIBUTED_EXEC=1`` and a configured ``REDIS_URL``.
    """
    return os.getenv("PYHARNESS_DISTRIBUTED_EXEC") == "1" and bool(os.getenv("REDIS_URL"))


def result_key(task_id: uuid.UUID | str) -> str:
    return f"{_RESULT_PREFIX}{task_id}"


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    """Recursively convert to JSON-safe primitives (pydantic-aware)."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        import base64

        return {"__kind__": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
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
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return str(value)


def _from_jsonable(value: Any) -> Any:
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


def serialize_task(task: ToolTask) -> str:
    return json.dumps(_jsonable(task.model_dump(mode="json")))


def deserialize_task(raw: str) -> ToolTask:
    return ToolTask.model_validate(_from_jsonable(json.loads(raw)))


def serialize_result(task_id: uuid.UUID, result: ToolResult) -> str:
    return json.dumps(_jsonable(ToolTaskResult(task_id=task_id, result=result).model_dump(mode="json")))


def deserialize_result(raw: str) -> ToolTaskResult:
    return ToolTaskResult.model_validate(_from_jsonable(json.loads(raw)))


# --------------------------------------------------------------------------- #
# Web-node side: enqueue + wait
# --------------------------------------------------------------------------- #
async def enqueue_tool_task(rclient: Any, task: ToolTask) -> None:
    """Push a :class:`ToolTask` onto the task queue (web-node side)."""
    await rclient.lpush(TASK_QUEUE_KEY, serialize_task(task))


async def await_tool_result(rclient: Any, task_id: uuid.UUID, timeout: float) -> ToolResult:
    """Block until a worker publishes the result, or return an error on timeout.

    Never hangs indefinitely: on ``BRPOP`` timeout (or cancel) a synthetic
    error :class:`ToolResult` is returned so the web node can fail cleanly.
    """
    key = result_key(task_id)
    try:
        # redis BRPOP returns (key, value) tuple or None on timeout.
        item = await rclient.brpop(key, timeout=timeout)
    except Exception as exc:  # connection errors, cancellation, etc.
        logger.warning("await_tool_result error for %s: %s", task_id, exc)
        return ToolResult(
            tool_name="?",
            status=ToolResultStatus.ERROR,
            error=f"distributed tool wait failed: {exc}",
        )

    if item is None:
        return ToolResult(
            tool_name="?",
            status=ToolResultStatus.ERROR,
            error="distributed tool execution timed out (no worker consumed it)",
        )

    try:
        envelope = deserialize_result(item[1])
    except Exception as exc:
        return ToolResult(tool_name="?", status=ToolResultStatus.ERROR, error=f"bad result envelope: {exc}")
    return envelope.result


# --------------------------------------------------------------------------- #
# Worker side: BRPOP loop
# --------------------------------------------------------------------------- #
async def process_one(rclient: Any, harness: Any, *, poll_timeout: float = 1.0) -> bool:
    """Consume a single task off the queue and execute it, if any.

    Returns ``True`` if a task was consumed and processed, ``False`` if the
    queue was empty within ``poll_timeout``. This is the deterministic,
    testable unit of worker behaviour; :func:`run_worker` loops over it.
    """
    item = await rclient.brpop(TASK_QUEUE_KEY, timeout=poll_timeout)
    if item is None:
        return False
    _raw_task = item[1]
    try:
        task = deserialize_task(_raw_task)
    except Exception as exc:  # noqa: BLE001
        logger.warning("worker dropped malformed task: %s", exc)
        return True

    result = await _execute_task(harness, task)
    try:
        await rclient.lpush(result_key(task.task_id), serialize_result(task.task_id, result))
    except Exception as exc:  # noqa: BLE001
        logger.warning("worker failed to write result for %s: %s", task.task_id, exc)
    return True


async def run_worker(rclient: Any, harness: Any, *, poll_timeout: float = 1.0) -> None:
    """Consume tasks off the queue and execute them on ``harness``.

    Runs forever until cancelled. For each task:
    1. Look up the registered tool executor via the ``execute_tool`` hook.
    2. Execute with the task's embedded :class:`ToolSpec` and arguments.
    3. Write the resulting :class:`ToolResult` back to the per-task result list.

    Failures never hang the web node: any exception is caught and turned into an
    error :class:`ToolResult` that is still published back.
    """
    while True:
        try:
            await process_one(rclient, harness, poll_timeout=poll_timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never crash the worker
            logger.warning("worker loop error: %s", exc, exc_info=True)


async def _execute_task(harness: Any, task: ToolTask) -> ToolResult:
    """Run one task through the harness's tool executor hook.

    Unlike the in-process engine's ``_settle`` (which isolates and *swallows*
    plugin exceptions to keep the agent loop alive), a compute worker **must**
    surface genuine tool failures back to the web node. We therefore await the
    raw hook results ourselves and treat any escaping exception as an error.
    """
    from pyharness.context import SessionContext
    from pyharness.schema import EventType, ToolStreamEvent

    started = time.monotonic()
    try:

        async def emitter(event: ToolStreamEvent) -> None:
            ctx = SessionContext(session_id=task.session_id)
            await harness.bus.aemit(EventType.TOOL_STREAM.value, context=ctx, event=event)

        ctx = SessionContext(session_id=task.session_id, tool_emitter=emitter)
        raw = harness.bus.pm.hook.execute_tool(context=ctx, tool=task.tool, arguments=task.arguments)
        # pluggy returns either a single awaitable or a collection of results.
        # Mirror _settle's normalization, but let exceptions escape to the
        # handler below so tool failures surface as error results.
        if inspect.isawaitable(raw):
            raw = (raw,)
        elif raw is None:
            raw = ()
        for value in raw:
            result = await value if inspect.isawaitable(value) else value
            if result is not None:
                return ToolResult.model_validate(result)
        return ToolResult(
            tool_name=task.tool.name,
            status=ToolResultStatus.ERROR,
            error="no executor plugin on worker",
        )
    except Exception as exc:  # noqa: BLE001 - must never hang the web node
        logger.warning("worker task %s failed: %s", task.task_id, exc, exc_info=True)
        return ToolResult(
            tool_name=task.tool.name,
            status=ToolResultStatus.ERROR,
            error=f"worker execution failed: {exc}",
            duration_seconds=time.monotonic() - started,
        )


__all__ = [
    "TASK_QUEUE_KEY",
    "distributed_exec_enabled",
    "result_key",
    "serialize_task",
    "deserialize_task",
    "serialize_result",
    "deserialize_result",
    "enqueue_tool_task",
    "await_tool_result",
    "process_one",
    "run_worker",
]
