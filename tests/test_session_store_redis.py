"""Tests for Redis session store plugin (distributed mode).

Uses fakeredis to avoid requiring a real Redis server.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure both `redis` and `fakeredis` are available before running these tests.
# redis is a [distributed] optional dependency; fakeredis is a test-only dep that
# avoids hitting a real Redis server. Skip the whole module if either is missing.
for _dep in ("redis", "fakeredis.aioredis"):
    try:
        __import__(_dep)
    except ImportError:
        pytest.skip(f"{_dep} not installed; skipping Redis session store tests", allow_module_level=True)

from pyharness.plugins.session_store_redis import RedisSessionStorePlugin
from pyharness.schema import (
    AgentConfig,
    HarnessConfig,
    LLMResponse,
    MemorySearchResult,
    Message,
    Role,
    SessionData,
    ToolCall,
    WorkflowPlan,
    WorkflowStep,
    StepStatus,
)
from pyharness import Harness
from pyharness.plugins.llm import entry as llm
from pyharness.context import SessionContext


def _make_fake_redis():
    """Create a fakeredis instance that mimics redis.asyncio.Redis."""
    import fakeredis.aioredis
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _harness_with_redis_store(fake_redis, *, auto_load: bool = False) -> tuple[Harness, RedisSessionStorePlugin]:
    """Build a Harness wired with the LLM plugin and a Redis session store (using fakeredis)."""
    h = Harness(config=HarnessConfig(auto_load_entry_points=auto_load))
    llm.clear()
    llm.use_dummy(models=("dummy",), plan=[LLMResponse(model="dummy", content="hi there")])
    h.register_plugin(llm)
    store = RedisSessionStorePlugin(redis_url="redis://fake")
    # Replace the internal redis client with fakeredis
    store._redis = fake_redis
    store._initialized = True
    h.register_plugin(store)
    h.initialize()
    return h, store


def _make_session(
    namespace: str = "test",
    memory: dict | None = None,
    messages: tuple[Message, ...] = (),
) -> SessionData:
    """Create a SessionData for testing (avoids frozen model issues)."""
    from datetime import datetime, timezone
    import uuid as _uuid
    return SessionData(
        session_id=_uuid.uuid4(),
        namespace=namespace,
        messages=messages,
        memory=memory or {},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture()
def fake_redis():
    """Provide a fresh fakeredis instance for each test."""
    return _make_fake_redis()


async def test_redis_save_and_load_session(fake_redis) -> None:
    """Test save_session and load_session round-trip with multimodal content."""
    store = RedisSessionStorePlugin(redis_url="redis://fake")
    store._redis = fake_redis
    store._initialized = True

    try:
        # Create a session with multimodal messages and metadata
        ctx = _make_session(
            namespace="test",
            memory={"user_pref": "dark_mode", "title": "Test Session"},
            messages=(
                Message(role=Role.USER, content="Hello", name=None),
                Message(
                    role=Role.ASSISTANT,
                    content="Hi there!",
                    name=None,
                    tool_calls=(
                        ToolCall(id="call_1", tool_name="python_exec", arguments={"code": "print(1)"}),
                    ),
                ),
                Message(role=Role.USER, content="How are you?", name=None),
            ),
        )

        await store.save_session(ctx)

        # Load it back
        loaded = await store.load_session(str(ctx.session_id))

        assert loaded is not None
        assert loaded.session_id == ctx.session_id
        assert loaded.namespace == "test"
        assert loaded.memory == {"user_pref": "dark_mode", "title": "Test Session"}
        assert len(loaded.messages) == 3
        assert loaded.messages[0].role == Role.USER
        assert loaded.messages[0].content == "Hello"
        assert loaded.messages[1].role == Role.ASSISTANT
        assert loaded.messages[1].content == "Hi there!"
        assert len(loaded.messages[1].tool_calls) == 1
        assert loaded.messages[1].tool_calls[0].tool_name == "python_exec"
        assert loaded.messages[2].role == Role.USER
        assert loaded.messages[2].content == "How are you?"
        assert loaded.message_count == 3
    finally:
        await store.teardown()


async def test_redis_list_sessions_ordered_by_timestamp(fake_redis) -> None:
    """Test list_sessions returns sessions ordered by updated_at desc with message_count."""
    store = RedisSessionStorePlugin(redis_url="redis://fake")
    store._redis = fake_redis
    store._initialized = True

    try:
        # Create multiple sessions with different update times
        for i in range(3):
            ctx = _make_session(
                namespace="default",
                memory={"index": i},
                messages=tuple(Message(role=Role.USER, content=f"msg {i}-{j}") for j in range(i + 1)),
            )
            await store.save_session(ctx)
            # Small delay to ensure different timestamps
            import asyncio
            await asyncio.sleep(0.01)

        sessions = await store.list_sessions(namespace="default", limit=10)

        assert len(sessions) == 3
        # Should be ordered by updated_at DESC (most recent first)
        # The last saved (index=2) should be first
        # Note: list_sessions returns lightweight sessions without memory dict
        # So we verify ordering by checking message_count and order
        assert sessions[0].message_count == 3  # index 2 has 3 messages
        assert sessions[1].message_count == 2  # index 1 has 2 messages
        assert sessions[2].message_count == 1  # index 0 has 1 message
    finally:
        await store.teardown()


async def test_redis_search_session(fake_redis) -> None:
    """Test search_session returns matching messages with snippets."""
    store = RedisSessionStorePlugin(redis_url="redis://fake")
    store._redis = fake_redis
    store._initialized = True

    try:
        ctx = _make_session(
            namespace="test",
            messages=(
                Message(role=Role.USER, content="Python is a great programming language"),
                Message(role=Role.ASSISTANT, content="Yes, Python is versatile"),
                Message(role=Role.USER, content="I also like Rust"),
            ),
        )
        await store.save_session(ctx)

        results = await store.search_session(str(ctx.session_id), query="Python", limit=10)

        assert len(results) >= 2
        for r in results:
            assert isinstance(r, MemorySearchResult)
            assert "Python" in r.content or "Python" in r.snippet
            assert r.session_id == str(ctx.session_id)
            assert r.rank >= 0
    finally:
        await store.teardown()


async def test_redis_delete_session(fake_redis) -> None:
    """Test delete_session removes session and returns True/False correctly."""
    store = RedisSessionStorePlugin(redis_url="redis://fake")
    store._redis = fake_redis
    store._initialized = True

    try:
        ctx = _make_session(
            namespace="test",
            messages=(Message(role=Role.USER, content="to delete"),),
        )
        await store.save_session(ctx)
        sid = str(ctx.session_id)

        # Delete existing
        deleted = await store.delete_session(sid)
        assert deleted is True

        # Verify gone
        loaded = await store.load_session(sid)
        assert loaded is None

        # Delete non-existing
        deleted = await store.delete_session("00000000-0000-0000-0000-000000000000")
        assert deleted is False
    finally:
        await store.teardown()


async def test_redis_rename_session(fake_redis) -> None:
    """Test rename_session updates title in metadata."""
    store = RedisSessionStorePlugin(redis_url="redis://fake")
    store._redis = fake_redis
    store._initialized = True

    try:
        ctx = _make_session(
            namespace="test",
            memory={"title": "Old Title"},
            messages=(Message(role=Role.USER, content="hello"),),
        )
        await store.save_session(ctx)
        sid = str(ctx.session_id)

        renamed = await store.rename_session(sid, "New Title")

        assert renamed is not None
        assert renamed.memory["title"] == "New Title"
        assert renamed.session_id == ctx.session_id
    finally:
        await store.teardown()


async def test_redis_clear_sessions(fake_redis) -> None:
    """Test clear_sessions removes all sessions in namespace."""
    store = RedisSessionStorePlugin(redis_url="redis://fake")
    store._redis = fake_redis
    store._initialized = True

    try:
        # Create sessions in the store's default namespace
        for i in range(3):
            ctx = _make_session(
                namespace="default",
                messages=(Message(role=Role.USER, content=f"msg {i}"),),
            )
            await store.save_session(ctx)

        count = await store.clear_sessions()
        assert count == 3

        sessions = await store.list_sessions(namespace="default")
        assert len(sessions) == 0
    finally:
        await store.teardown()


async def test_redis_save_and_load_plan(fake_redis) -> None:
    """Test save_plan and load_plan round-trip."""
    store = RedisSessionStorePlugin(redis_url="redis://fake")
    store._redis = fake_redis
    store._initialized = True

    try:
        sid = "00000000-0000-0000-0000-000000000001"
        plan = WorkflowPlan(
            task="Test plan",
            steps=(
                WorkflowStep(id="s1", title="Step 1", description="Do something", status=StepStatus.COMPLETED),
                WorkflowStep(id="s2", title="Step 2", description="Do more", status=StepStatus.PENDING),
            ),
            status="running",
            progress=0.5,
        )
        await store.save_plan(plan, sid)

        loaded = await store.load_plan(str(plan.plan_id))

        assert loaded is not None
        assert loaded.plan_id == plan.plan_id
        assert loaded.task == "Test plan"
        assert loaded.status == "running"
        assert loaded.progress == 0.5
        assert len(loaded.steps) == 2
        assert loaded.steps[0].id == "s1"
        assert loaded.steps[0].status == StepStatus.COMPLETED
        assert loaded.steps[1].id == "s2"
        assert loaded.steps[1].status == StepStatus.PENDING
    finally:
        await store.teardown()


async def test_redis_list_plans(fake_redis) -> None:
    """Test list_plans returns all plans for a session ordered by created_at."""
    store = RedisSessionStorePlugin(redis_url="redis://fake")
    store._redis = fake_redis
    store._initialized = True

    try:
        sid = "00000000-0000-0000-0000-000000000002"
        for i in range(3):
            plan = WorkflowPlan(
                task=f"Plan {i}",
                steps=(WorkflowStep(id=f"s{i}", title=f"Step {i}", description="", status=StepStatus.PENDING),),
                status="running",
                progress=0.0,
            )
            await store.save_plan(plan, sid)
            import asyncio
            await asyncio.sleep(0.01)

        plans = await store.list_plans(sid)

        assert len(plans) == 3
        # Ordered by created_at
        assert plans[0].task == "Plan 0"
        assert plans[1].task == "Plan 1"
        assert plans[2].task == "Plan 2"
    finally:
        await store.teardown()


async def _factory_harness(*, redis_url: str | None) -> Harness:
    """Build a Harness via ``factory.build_harness`` for a given REDIS_URL setting.

    The Redis plugin's real ``initialize`` is monkeypatched inside the caller to
    avoid hitting a live Redis server; this helper just performs the build.
    """
    from pyharness import factory

    if redis_url is None:
        os.environ.pop("REDIS_URL", None)
    else:
        os.environ["REDIS_URL"] = redis_url

    return factory.build_harness(model="dummy", provider="dummy")


async def test_factory_routes_to_redis_when_redisd_set(fake_redis, monkeypatch) -> None:
    """Factory should use RedisSessionStorePlugin when REDIS_URL is set."""
    # Only degrade to SQLite on ImportError; a configured-but-unreachable store
    # must remain Redis (routing decision is independent of connectivity).
    async def mock_initialize(self) -> None:
        self._redis = fake_redis
        self._initialized = True

    monkeypatch.setattr(RedisSessionStorePlugin, "initialize", mock_initialize)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    try:
        from pyharness import factory
        import importlib
        importlib.reload(factory)  # pick up the env var for _session_store()

        harness = factory.build_harness(model="dummy", provider="dummy")
        redis_plugin = any(
            isinstance(p, RedisSessionStorePlugin)
            for p in harness.bus.pm.get_plugins()
        )
        assert redis_plugin, "RedisSessionStorePlugin not registered"
    finally:
        monkeypatch.delenv("REDIS_URL", raising=False)


async def test_factory_routes_to_sqlite_when_no_redis_url(monkeypatch) -> None:
    """Factory should use SQLiteSessionStorePlugin when REDIS_URL is not set."""
    monkeypatch.delenv("REDIS_URL", raising=False)

    import importlib
    from pyharness import factory
    importlib.reload(factory)

    try:
        harness = factory.build_harness(model="dummy", provider="dummy")
        from pyharness.plugins.session_store import SQLiteSessionStorePlugin
        sqlite_plugin = any(
            isinstance(p, SQLiteSessionStorePlugin)
            for p in harness.bus.pm.get_plugins()
        )
        assert sqlite_plugin, "SQLiteSessionStorePlugin not registered"
    finally:
        importlib.reload(factory)


async def test_redis_initialize_fail_fast_on_connection_error(monkeypatch) -> None:
    """initialize() must raise (fail-fast), never degrade, when Redis is unreachable."""
    store = RedisSessionStorePlugin(redis_url="redis://localhost:6379/0")

    class _Unreachable:
        async def ping(self):
            from redis.exceptions import ConnectionError
            raise ConnectionError("connection refused")

    import redis.asyncio as real_redis_asyncio
    monkeypatch.setattr(real_redis_asyncio, "from_url", lambda *a, **k: _Unreachable())

    with pytest.raises(RuntimeError, match="fail-fast"):
        await store.initialize()

    assert store._initialized is False, "must not mark as initialized after failed ping"


async def test_redis_ttl_refresh_on_write(fake_redis) -> None:
    """Test that TTL is refreshed on each write."""
    store = RedisSessionStorePlugin(redis_url="redis://fake", ttl_days=1)
    store._redis = fake_redis
    store._initialized = True

    try:
        ctx = _make_session(
            namespace="test",
            messages=(Message(role=Role.USER, content="initial"),),
        )
        await store.save_session(ctx)
        sid = str(ctx.session_id)

        # Check TTL is set
        key = store._session_key(sid)
        ttl = await fake_redis.ttl(key)
        assert ttl > 0
        assert ttl <= 86400  # 1 day in seconds

        # Update session
        ctx = _make_session(
            namespace="test",
            messages=(Message(role=Role.USER, content="updated"),),
        )
        ctx = ctx.model_copy(update={"session_id": sid})
        await store.save_session(ctx)

        # TTL should be refreshed
        ttl2 = await fake_redis.ttl(key)
        assert ttl2 > 0
        assert ttl2 <= 86400
    finally:
        await store.teardown()


async def test_redis_memory_search_tool(fake_redis) -> None:
    """Test memory_search tool execution via Redis store."""
    from pyharness.schema import ToolResult, ToolResultStatus, ToolSpec, ToolArg

    store = RedisSessionStorePlugin(redis_url="redis://fake")
    store._redis = fake_redis
    store._initialized = True

    try:
        ctx = _make_session(
            namespace="test",
            messages=(
                Message(role=Role.USER, content="Remember: buy milk and eggs"),
                Message(role=Role.ASSISTANT, content="Got it"),
            ),
        )
        await store.save_session(ctx)

        tool = ToolSpec(
            name="memory_search",
            description="Search memory",
            parameters=(ToolArg(name="query", type="string", description="Query", required=True),),
        )

        result = await store.execute_tool(ctx, tool, {"query": "milk", "limit": 5})

        assert result.status == ToolResultStatus.OK
        assert result.output["count"] >= 1
        assert "milk" in str(result.output["results"][0]["content"]).lower()
    finally:
        await store.teardown()


if __name__ == "__main__":
    import asyncio
    import tempfile
    from pathlib import Path

    async def run_all():
        fake_redis = _make_fake_redis()
        await test_redis_save_and_load_session(fake_redis)
        print("PASS: test_redis_save_and_load_session")
        fake_redis = _make_fake_redis()
        await test_redis_list_sessions_ordered_by_timestamp(fake_redis)
        print("PASS: test_redis_list_sessions_ordered_by_timestamp")
        fake_redis = _make_fake_redis()
        await test_redis_search_session(fake_redis)
        print("PASS: test_redis_search_session")
        fake_redis = _make_fake_redis()
        await test_redis_delete_session(fake_redis)
        print("PASS: test_redis_delete_session")
        fake_redis = _make_fake_redis()
        await test_redis_rename_session(fake_redis)
        print("PASS: test_redis_rename_session")
        fake_redis = _make_fake_redis()
        await test_redis_clear_sessions(fake_redis)
        print("PASS: test_redis_clear_sessions")
        fake_redis = _make_fake_redis()
        await test_redis_save_and_load_plan(fake_redis)
        print("PASS: test_redis_save_and_load_plan")
        fake_redis = _make_fake_redis()
        await test_redis_list_plans(fake_redis)
        print("PASS: test_redis_list_plans")
        fake_redis = _make_fake_redis()
        await test_redis_ttl_refresh_on_write(fake_redis)
        print("PASS: test_redis_ttl_refresh_on_write")
        fake_redis = _make_fake_redis()
        await test_redis_memory_search_tool(fake_redis)
        print("PASS: test_redis_memory_search_tool")

    asyncio.run(run_all())