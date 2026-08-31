"""Redis-backed session store plugin for PyHarness (distributed mode).

Implements the same ``SessionStoreHooks`` as :class:`SQLiteSessionStorePlugin`
but persists to Redis for horizontal scalability across multiple harness instances.

Data Model
----------
* ``session:<ns>:<session_id>`` -> Hash (session metadata: created_at, updated_at, namespace, memory_json)
* ``session:<ns>:<session_id>:messages`` -> List (JSON-serialized Messages, left-push = append)
* ``session:<ns>:<session_id>:fts`` -> List (JSON for FTS: [session_id, role, content])
* ``user:<user_id>:sessions:<ns>`` -> Sorted Set (member=session_id, score=updated_at timestamp)

TTL
---
Keys expire after 30 days of inactivity (configurable via ``ttl_days``).
Each write refreshes the TTL.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from pluggy import HookimplMarker

from pyharness.schema import (
    MemorySearchResult,
    Message,
    SessionData,
    ToolCall,
    WorkflowPlan,
    WorkflowStep,
)
from pyharness.specs import AgentHooks

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")


def _utcnow() -> float:
    return time.time()


def _utcnow_dt() -> datetime:
    return datetime.now(timezone.utc)


class RedisSessionStorePlugin:
    """Async Redis persistence for PyHarness sessions.

    Parameters
    ----------
    redis_url:
        Redis connection URL (e.g. ``redis://localhost:6379/0``).
        If not provided, reads from ``REDIS_URL`` environment variable.
    namespace:
        Logical namespace to isolate sessions (default: "default").
        Used as a prefix for all Redis keys.
    ttl_days:
        Time-to-live in days for session keys. Default 30 days.
        TTL is refreshed on every write.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        namespace: str = "default",
        ttl_days: int = 30,
    ) -> None:
        import os

        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.namespace = namespace
        self.ttl_seconds = ttl_days * 86400
        self._redis: Any = None  # redis.asyncio.Redis | None
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def initialize(self) -> None:
        """Create Redis connection pool and verify connectivity.

        Fail-fast semantics:
        - ``ImportError`` (redis package not installed) -> re-raised with an
          actionable install hint.
        - Any connection failure during ``ping()`` (e.g. ``ConnectionError``,
          ``TimeoutError``) -> the exception propagates uncaught so the service
          refuses to start. NEVER fall back to SQLite here: multiple nodes must
          not silently split-brain onto different stores.
        """
        if self._initialized:
            return

        try:
            import redis.asyncio as redis
        except ImportError as exc:
            raise ImportError(
                "RedisSessionStorePlugin requires 'redis[hiredis]'. "
                "Install with: pip install pyharness[distributed]"
            ) from exc

        if not self.redis_url:
            raise ValueError(
                "RedisSessionStorePlugin requires a Redis URL. "
                "Pass redis_url= or set REDIS_URL environment variable."
            )

        self._redis = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        # Verify connectivity. Any failure here is fail-fast: it propagates up
        # and prevents the harness from starting against an unhealthy Redis.
        # SQLite must NOT be substituted, or nodes would split-brain.
        try:
            await self._redis.ping()
        except Exception as exc:
            raise RuntimeError(
                "Redis connection failed (fail-fast): cannot reach "
                f"{self.redis_url!r}. Refusing to fall back to SQLite to avoid "
                "data split-brain across nodes. Check the Redis cluster state."
            ) from exc
        self._initialized = True
        logger.info("Redis session store initialized at %s (ns=%s)", self.redis_url, self.namespace)

    async def teardown(self) -> None:
        """Close Redis connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            self._initialized = False
            logger.info("Redis session store torn down")

    def _session_key(self, session_id: str) -> str:
        return f"session:{self.namespace}:{session_id}"

    def _messages_key(self, session_id: str) -> str:
        return f"session:{self.namespace}:{session_id}:messages"

    def _fts_key(self, session_id: str) -> str:
        return f"session:{self.namespace}:{session_id}:fts"

    def _sessions_index_key(self, namespace: str | None = None) -> str:
        ns = namespace if namespace is not None else self.namespace
        return f"user:default:sessions:{ns}"

    def _plans_key(self, session_id: str) -> str:
        return f"session:{self.namespace}:{session_id}:plans"

    def _plan_key(self, plan_id: str) -> str:
        return f"plan:{self.namespace}:{plan_id}"

    async def _refresh_ttl(self, *keys: str) -> None:
        """Refresh TTL on multiple keys."""
        if self._redis is None:
            return
        try:
            await self._redis.expire(*keys, time=self.ttl_seconds)
        except Exception as exc:
            logger.warning("Failed to refresh TTL for %s: %s", keys, exc)

    # ------------------------------------------------------------------ #
    # Hook implementations
    # ------------------------------------------------------------------ #
    @hookimpl
    async def load_session(self, session_id: str) -> SessionData | None:
        """Restore a session's pure data from Redis by its string id."""
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return None

        try:
            key = self._session_key(session_id)
            data = await self._redis.hgetall(key)
            if not data:
                return None

            messages = await self._load_messages(session_id)

            return SessionData(
                session_id=uuid.UUID(data["id"]),
                namespace=data.get("namespace", self.namespace),
                messages=tuple(messages),
                memory=json.loads(data.get("metadata", "{}")),
                created_at=datetime.fromtimestamp(
                    float(data["created_at"]), tz=timezone.utc
                ),
                message_count=len(messages),
            )
        except Exception as exc:
            logger.warning("load_session(%s) failed: %s", session_id, exc)
            return None

    @hookimpl
    async def list_sessions(
        self, namespace: str = "default", limit: int = 50, offset: int = 0
    ) -> list[SessionData]:
        """List persisted sessions ordered by most recent update (ZREVRANGE)."""
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return []

        try:
            index_key = self._sessions_index_key(namespace)
            # ZREVRANGE gets highest scores first (most recent updated_at)
            session_ids = await self._redis.zrevrange(
                index_key, offset, offset + limit - 1
            )
            rows: list[SessionData] = []
            for sid in session_ids:
                key = self._session_key(sid)
                data = await self._redis.hgetall(key)
                if not data:
                    continue
                # Count messages (LLLEN on messages list)
                msg_count = await self._redis.llen(self._messages_key(sid))
                rows.append(
                    SessionData(
                        session_id=uuid.UUID(data["id"]),
                        namespace=data.get("namespace", self.namespace),
                        messages=(),
                        message_count=msg_count,
                        created_at=datetime.fromtimestamp(
                            float(data["created_at"]), tz=timezone.utc
                        ),
                    )
                )
            return rows
        except Exception as exc:
            logger.warning("list_sessions failed: %s", exc)
            return []

    @hookimpl
    async def save_session(self, session: Any) -> None:
        """Persist the current session's pure data to Redis."""
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return

        try:
            sid = str(session.session_id)
            now = _utcnow()
            metadata_json = json.dumps(dict(session.memory), ensure_ascii=False)

            key = self._session_key(sid)
            msg_key = self._messages_key(sid)
            fts_key = self._fts_key(sid)
            index_key = self._sessions_index_key(session.namespace)

            async with self._redis.pipeline(transaction=True) as pipe:
                # Session metadata hash
                pipe.hset(
                    key,
                    mapping={
                        "id": sid,
                        "created_at": str(session.created_at.timestamp()),
                        "updated_at": str(now),
                        "namespace": session.namespace,
                        "metadata": metadata_json,
                    },
                )
                # Messages list: delete old and rebuild
                pipe.delete(msg_key)
                pipe.delete(fts_key)
                if session.messages:
                    msg_pairs = self._iter_message_rows(sid, session.messages)
                    fts_pairs = self._iter_fts_rows(sid, session.messages)
                    # Redis LPUSH prepends; we want chronological order so we reverse
                    for pair in reversed(msg_pairs):
                        pipe.lpush(msg_key, json.dumps(pair, ensure_ascii=False))
                    for pair in reversed(fts_pairs):
                        pipe.lpush(fts_key, json.dumps(pair, ensure_ascii=False))
                # Sorted set index for list_sessions (score = updated_at)
                pipe.zadd(index_key, {sid: now})
                # Refresh TTL on all keys
                pipe.expire(key, self.ttl_seconds)
                pipe.expire(msg_key, self.ttl_seconds)
                pipe.expire(fts_key, self.ttl_seconds)
                pipe.expire(index_key, self.ttl_seconds)
                await pipe.execute()
        except Exception as exc:
            logger.warning("save_session(%s) failed: %s", getattr(session, "session_id", "?"), exc)

    @hookimpl
    async def search_session(
        self, session_id: str, query: str, limit: int = 10
    ) -> list[MemorySearchResult]:
        """Simple in-Redis full-text search (scan FTS list + filter in Python).

        Note: For production high-scale FTS, consider Redisearch (RediSearch module).
        This implementation does a linear scan of the session's FTS list,
        which is acceptable for typical session sizes (<10k messages).
        """
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return []

        try:
            safe_query = self._sanitize_fts_query(query).lower()
            fts_key = self._fts_key(session_id)
            rows: list[MemorySearchResult] = []

            # LRANGE to get all FTS entries (typically small per session)
            fts_entries = await self._redis.lrange(fts_key, 0, -1)
            for entry_json in fts_entries:
                try:
                    entry = json.loads(entry_json)
                    # entry = [session_id, role, content]
                    content = entry[2]
                    if safe_query.strip('"') in content.lower():
                        # Simple rank: position in list (earlier = better for demo)
                        # In production, use Redisearch for BM25
                        rank = float(fts_entries.index(entry_json))
                        rows.append(
                            MemorySearchResult(
                                session_id=entry[0],
                                role=entry[1],
                                content=content,
                                snippet=self._make_snippet(content, safe_query.strip('"')),
                                rank=rank,
                            )
                        )
                except Exception:
                    continue

            # Sort by rank (lower = better)
            rows.sort(key=lambda r: r.rank)
            return rows[:limit]
        except Exception as exc:
            logger.warning("search_session(%s, %r) failed: %s", session_id, query, exc)
            return []

    @hookimpl
    async def delete_session(self, session_id: str) -> bool:
        """Delete a persisted session and all its data from Redis."""
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return False

        try:
            key = self._session_key(session_id)
            data = await self._redis.hgetall(key)
            if not data:
                return False

            namespace = data.get("namespace", self.namespace)
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.delete(key)
                pipe.delete(self._messages_key(session_id))
                pipe.delete(self._fts_key(session_id))
                pipe.zrem(self._sessions_index_key(namespace), session_id)
                await pipe.execute()
            return True
        except Exception as exc:
            logger.warning("delete_session(%s) failed: %s", session_id, exc)
            return False

    @hookimpl
    async def rename_session(self, session_id: str, title: str) -> SessionData | None:
        """Rename a persisted session by updating its metadata.title."""
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return None

        try:
            key = self._session_key(session_id)
            data = await self._redis.hgetall(key)
            if not data:
                return None

            metadata = json.loads(data.get("metadata", "{}"))
            metadata["title"] = title
            now = _utcnow()
            namespace = data.get("namespace", self.namespace)

            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.hset(key, "metadata", json.dumps(metadata, ensure_ascii=False))
                pipe.hset(key, "updated_at", str(now))
                pipe.zadd(self._sessions_index_key(namespace), {session_id: now})
                await pipe.execute()

            messages = await self._load_messages(session_id)
            return SessionData(
                session_id=uuid.UUID(session_id),
                namespace=data.get("namespace", self.namespace),
                messages=tuple(messages),
                memory=metadata,
                created_at=datetime.fromtimestamp(float(data["created_at"]), tz=timezone.utc),
                message_count=len(messages),
            )
        except Exception as exc:
            logger.warning("rename_session(%s) failed: %s", session_id, exc)
            return None

    @hookimpl
    async def clear_sessions(self) -> int:
        """Delete all persisted sessions in this namespace from Redis."""
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return 0

        try:
            index_key = self._sessions_index_key()
            session_ids = await self._redis.zrange(index_key, 0, -1)
            count = len(session_ids)

            if count == 0:
                return 0

            async with self._redis.pipeline(transaction=True) as pipe:
                for sid in session_ids:
                    pipe.delete(self._session_key(sid))
                    pipe.delete(self._messages_key(sid))
                    pipe.delete(self._fts_key(sid))
                    pipe.delete(self._plans_key(sid))
                pipe.delete(index_key)
                await pipe.execute()
            return count
        except Exception as exc:
            logger.warning("clear_sessions failed: %s", exc)
            return 0

    @hookimpl
    async def save_plan(self, plan: WorkflowPlan, session_id: str) -> None:
        """Persist a WorkflowPlan to Redis."""
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return

        try:
            steps_json = json.dumps(
                [s.model_dump(mode="json") for s in plan.steps], ensure_ascii=False
            )
            now = _utcnow()

            plan_key = self._plan_key(str(plan.plan_id))
            session_plans_key = self._plans_key(session_id)

            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.hset(
                    plan_key,
                    mapping={
                        "plan_id": str(plan.plan_id),
                        "session_id": session_id,
                        "goal": plan.task,
                        "status": plan.status,
                        "progress": str(plan.progress),
                        "steps_json": steps_json,
                        "created_at": str(plan.created_at.timestamp()),
                        "updated_at": str(now),
                    },
                )
                pipe.expire(plan_key, self.ttl_seconds)
                # Index plan_id in session's plan set
                pipe.sadd(session_plans_key, str(plan.plan_id))
                pipe.expire(session_plans_key, self.ttl_seconds)
                await pipe.execute()
        except Exception as exc:
            logger.warning("save_plan(%s) failed: %s", getattr(plan, "plan_id", "?"), exc)

    @hookimpl
    async def load_plan(self, plan_id: str) -> WorkflowPlan | None:
        """Load a persisted WorkflowPlan from Redis."""
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return None

        try:
            plan_key = self._plan_key(plan_id)
            data = await self._redis.hgetall(plan_key)
            if not data:
                return None

            steps = []
            for step_data in json.loads(data["steps_json"]):
                steps.append(WorkflowStep.model_validate(step_data))

            return WorkflowPlan(
                plan_id=uuid.UUID(data["plan_id"]),
                task=data["goal"],
                steps=tuple(steps),
                status=data["status"],
                progress=float(data["progress"]),
                created_at=datetime.fromtimestamp(
                    float(data["created_at"]), tz=timezone.utc
                ),
            )
        except Exception as exc:
            logger.warning("load_plan(%s) failed: %s", plan_id, exc)
            return None

    @hookimpl
    async def list_plans(self, session_id: str) -> list[WorkflowPlan]:
        """List all plans for a given session."""
        if not self._initialized:
            await self.initialize()
        if self._redis is None:
            return []

        try:
            session_plans_key = self._plans_key(session_id)
            plan_ids = await self._redis.smembers(session_plans_key)
            plans: list[WorkflowPlan] = []

            for pid in plan_ids:
                plan = await self.load_plan(pid)
                if plan:
                    plans.append(plan)

            # Sort by created_at
            plans.sort(key=lambda p: p.created_at)
            return plans
        except Exception as exc:
            logger.warning("list_plans(%s) failed: %s", session_id, exc)
            return []

    # ------------------------------------------------------------------ #
    # Tools (memory_search) — same as SQLite plugin
    # ------------------------------------------------------------------ #
    def _memory_search_spec(self):
        from pyharness.schema import ToolArg, ToolSpec
        return ToolSpec(
            name="memory_search",
            description=(
                "在历史对话记忆中搜索相关内容。"
                "使用全文检索找到最相关的过往消息，返回摘要片段和相关性评分。"
                "当你需要回忆之前讨论过的内容时使用此工具。"
            ),
            parameters=(
                ToolArg(name="query", type="string", description="搜索关键词", required=True),
                ToolArg(name="limit", type="integer", description="最大返回条数（默认 5）", required=False),
            ),
            timeout_seconds=10.0,
        )

    @hookimpl
    def get_tool_specs(self, context: Any) -> tuple:
        from pyharness.context import SessionContext
        if isinstance(context, SessionContext):
            return (self._memory_search_spec(),)
        return (self._memory_search_spec(),)

    @hookimpl
    async def execute_tool(
        self, context: Any, tool: Any, arguments: dict[str, object]
    ) -> Any | None:
        from pyharness.schema import ToolResult, ToolResultStatus

        if tool.name != "memory_search":
            return None
        query = str(arguments.get("query", ""))
        limit = int(arguments.get("limit", 5))
        if not query:
            return ToolResult(
                tool_name="memory_search",
                status=ToolResultStatus.ERROR,
                error="缺少 'query' 参数。",
                output={},
            )

        # Accept both SessionContext and SessionData (both have session_id)
        session_id = getattr(context, 'session_id', None)
        if session_id is None:
            return ToolResult(
                tool_name="memory_search",
                status=ToolResultStatus.ERROR,
                error="Invalid context for memory search.",
                output={},
            )

        session_id = str(session_id)
        results = await self.search_session(session_id=session_id, query=query, limit=limit)
        output = {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "role": r.role,
                    "content": r.content,
                    "snippet": r.snippet,
                    "rank": r.rank,
                }
                for r in results
            ],
        }
        return ToolResult(tool_name="memory_search", status=ToolResultStatus.OK, output=output)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _load_messages(self, session_id: str) -> list[Message]:
        """Load all messages for session_id from Redis list."""
        msg_key = self._messages_key(session_id)
        rows: list[Message] = []
        try:
            msg_jsons = await self._redis.lrange(msg_key, 0, -1)
            for msg_json in msg_jsons:
                try:
                    msg_data = json.loads(msg_json)
                    # msg_data = [session_id, role, content, name, tool_calls_json]
                    tool_calls = json.loads(msg_data[4] or "[]")
                    rows.append(
                        Message(
                            role=msg_data[1],
                            content=msg_data[2],
                            name=msg_data[3] if msg_data[3] else None,
                            tool_calls=tuple(ToolCall(**tc) for tc in tool_calls),
                        )
                    )
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("_load_messages(%s) failed: %s", session_id, exc)
        return rows

    def _iter_message_rows(self, session_id: str, messages: Sequence[Message]) -> Sequence[list]:
        """Yield [session_id, role, content, name, tool_calls_json] for Redis list."""
        result: list[list] = []
        for msg in messages:
            tool_calls_json = json.dumps(
                [tc.model_dump(mode="json") for tc in msg.tool_calls], ensure_ascii=False
            )
            result.append([session_id, msg.role, msg.content, msg.name or "", tool_calls_json])
        return result

    def _iter_fts_rows(self, session_id: str, messages: Sequence[Message]) -> Sequence[list]:
        """Yield [session_id, role, content] for FTS list."""
        result: list[list] = []
        for msg in messages:
            result.append([session_id, msg.role, msg.content])
        return result

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize query for simple substring matching (not FTS5 syntax)."""
        return query.replace('"', '""')

    @staticmethod
    def _make_snippet(content: str, query: str, context_chars: int = 60) -> str:
        """Generate a highlighted snippet around the first match."""
        idx = content.lower().find(query.lower())
        if idx == -1:
            return content[:context_chars] + ("..." if len(content) > context_chars else "")
        start = max(0, idx - context_chars // 2)
        end = min(len(content), idx + len(query) + context_chars // 2)
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        return snippet


__all__ = ["RedisSessionStorePlugin"]