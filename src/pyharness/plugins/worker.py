"""Standalone distributed-execution worker.

A ``pyharness worker`` process consumes :class:`ToolTask` messages off the Redis
task queue and executes them locally (via the same ``execute_tool`` hook as the
web node), streaming intermediate ``tool.stream`` events back over the batch-2
Redis Pub/Sub event bus.

The worker wires ONLY the tool-executor plugins plus the Redis event bus — it
deliberately does **not** load the web UI, session store, or guard-approval, so
a worker is a pure compute unit.
"""

from __future__ import annotations

import logging
import os

from pyharness import Harness
from pyharness.distributed import TASK_QUEUE_KEY, run_worker
from pyharness.plugins.llm import entry as llm
from pyharness.plugins.tool_audio import AudioToolPlugin
from pyharness.plugins.tool_fs import FileSystemPlugin
from pyharness.plugins.tool_python_exec import PythonExecPlugin
from pyharness.plugins.tool_subagent import SubagentToolPlugin
from pyharness.plugins.tool_web import WebPlugin
from pyharness.plugins.workflow import WorkflowPlugin
from pyharness.schema import HarnessConfig

logger = logging.getLogger(__name__)


def build_worker_harness() -> Harness:
    """Build a compute-only Harness wired with tool executors + Redis event bus.

    Raises ``RuntimeError`` when ``REDIS_URL`` is not configured, since a worker
    is meaningless without the shared queue.
    """
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError(
            "pyharness worker requires REDIS_URL to be set (shared task queue)."
        )

    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    # Tool executor plugins only — compute units, no UI/session/store.
    h.register_plugin(FileSystemPlugin())
    h.register_plugin(WebPlugin())
    h.register_plugin(WorkflowPlugin())
    h.register_plugin(SubagentToolPlugin())
    h.register_plugin(PythonExecPlugin())
    h.register_plugin(AudioToolPlugin())
    # LLM provider is irrelevant to tool execution, but keep a dummy available
    # so tool plugins that negotiate/… don't blow up.
    llm.clear()
    llm.use_dummy(models=("dummy",))

    # Redis event bus so tool.stream events stream back to the web node.
    try:
        from pyharness.plugins.event_bus_redis import RedisEventBusPlugin

        h.register_plugin(RedisEventBusPlugin(redis_url=redis_url))
    except ImportError:
        logger.warning("redis library not found; worker streaming disabled")

    h.initialize()
    return h


async def run_worker_forever(*, poll_timeout: float = 1.0) -> None:
    """Open a Redis client and consume tasks until cancelled (Ctrl-C)."""
    import redis.asyncio as redis

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL is required to run a worker.")

    harness = build_worker_harness()
    client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    logger.info("worker consuming from queue '%s'", TASK_QUEUE_KEY)
    try:
        await run_worker(client, harness, poll_timeout=poll_timeout)
    finally:
        harness.shutdown()
        await client.aclose()


__all__ = ["build_worker_harness", "run_worker_forever"]
