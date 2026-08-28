"""SQLite-backed session store plugin for PyHarness.

Implements ``load_session`` / ``save_session`` hooks so the engine can
persist and resume agent sessions across process restarts, plus a
``search_session`` hook backed by SQLite FTS5 for full-text memory recall.

Design
------
* Uses ``aiosqlite`` for fully asynchronous SQLite access.
* Normalizes messages into a dedicated ``messages`` table for queryability.
* Serializes Pydantic models via ``model_dump(mode='json')`` and restores
  with ``model_validate``.
* Persists/loads the pure :class:`~pyharness.schema.SessionData` DTO, never
  a live :class:`~pyharness.context.SessionContext`. The engine is
  responsible for wrapping loaded data in a fresh context.
* Maintains an FTS5 virtual table for low-latency full-text search with
  BM25 ranking and highlighted snippets.
* Best-effort persistence: failures are swallowed so the agent loop never
  crashes because of storage issues.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from pluggy import HookimplMarker

from pyharness.schema import MemorySearchResult, Message, SessionData, ToolCall, ToolResult, ToolResultStatus, ToolSpec, ToolArg, WorkflowPlan
from pyharness.specs import AgentHooks

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")

_CREATE_SESSIONS = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    namespace   TEXT NOT NULL DEFAULT 'default',
    metadata    TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_MESSAGES = """\
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    name        TEXT,
    tool_calls  TEXT NOT NULL DEFAULT '[]'
);
"""

_CREATE_FTS = """\
CREATE VIRTUAL TABLE IF NOT EXISTS fts_messages USING fts5(
    session_id UNINDEXED,
    role UNINDEXED,
    content,
    tokenize='unicode61'
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id);
"""

_CREATE_PLANS = """\
CREATE TABLE IF NOT EXISTS plans (
    plan_id    TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    goal       TEXT NOT NULL,
    status     TEXT NOT NULL,
    progress   REAL NOT NULL DEFAULT 0.0,
    steps_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


def _utcnow() -> float:
    return time.time()


class SQLiteSessionStorePlugin:
    """Async SQLite persistence for PyHarness sessions.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Defaults to ``pyharness_sessions.db``
        in the current working directory.

    Notes
    -----
    This plugin lazily initializes the database connection on first use, so
    callers do not need to invoke :meth:`initialize` manually unless they
    want explicit lifecycle control.
    """

    def __init__(self, db_path: str = "pyharness_sessions.db") -> None:
        self.db_path = db_path
        self._db: Any = None  # aiosqlite.Connection | None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def initialize(self) -> None:
        """Open the database connection and ensure schema exists."""
        try:
            import aiosqlite
        except ImportError as exc:
            raise ImportError(
                "SQLiteSessionStorePlugin requires 'aiosqlite'. "
                "Install with: pip install pyharness[memory]"
            ) from exc

        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
            # Concurrency hardening: WAL allows concurrent readers with one
            # writer; busy_timeout avoids immediate "database is locked".
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA busy_timeout=5000")
            await self._db.executescript(_CREATE_SESSIONS)
            await self._db.executescript(_CREATE_MESSAGES)
            await self._db.executescript(_CREATE_FTS)
            await self._db.executescript(_CREATE_INDEX)
            await self._db.executescript(_CREATE_PLANS)
            await self._db.commit()
            logger.info("Session store initialized at %s", self.db_path)

    async def teardown(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("Session store torn down")

    # ------------------------------------------------------------------ #
    # Hook implementations
    # ------------------------------------------------------------------ #
    @hookimpl
    async def load_session(self, session_id: str) -> SessionData | None:
        """Restore a session's pure data from the database by its string id.

        Returns a :class:`SessionData` DTO (pure data, no runtime state), or
        ``None`` if the session is unknown. The engine is responsible for
        wrapping this in a fresh :class:`~pyharness.context.SessionContext`.
        """
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return None

        try:
            import datetime
            import uuid

            async with self._db.execute(
                "SELECT id, created_at, updated_at, namespace, metadata FROM sessions WHERE id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None

            session_row = dict(row)
            messages = await self._load_messages(session_id)

            return SessionData(
                session_id=uuid.UUID(session_row["id"]),
                namespace=session_row["namespace"],
                messages=tuple(messages),
                memory=json.loads(session_row.get("metadata") or "{}"),
                created_at=datetime.datetime.fromtimestamp(
                    session_row["created_at"], tz=datetime.timezone.utc
                ),
            )
        except Exception as exc:
            logger.warning("load_session(%s) failed: %s", session_id, exc)
            return None

    @hookimpl
    async def list_sessions(self, namespace: str = "default", limit: int = 50, offset: int = 0) -> list[SessionData]:
        """List persisted sessions ordered by most recent update.

        Lightweight: a single ``LEFT JOIN COUNT`` query fills ``message_count``
        and message bodies stay empty (``messages=()``) — no N+1 message loads
        and no conversation content leaks into list responses. Use
        :meth:`load_session` for full transcripts.
        """
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return []

        try:
            rows: list[SessionData] = []
            sql = """\
                SELECT s.id          AS id,
                       s.created_at  AS created_at,
                       s.namespace   AS namespace,
                       COUNT(m.id)   AS message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                WHERE s.namespace = ?
                GROUP BY s.id, s.created_at, s.namespace
                ORDER BY s.updated_at DESC
                LIMIT ? OFFSET ?
                """
            async with self._db.execute(sql, (namespace, limit, offset)) as cursor:
                async for row in cursor:
                    rows.append(
                        SessionData(
                            session_id=uuid.UUID(row["id"]),
                            namespace=row["namespace"],
                            messages=(),
                            message_count=int(row["message_count"]),
                            created_at=datetime.fromtimestamp(
                                row["created_at"], tz=timezone.utc
                            ),
                        )
                    )
            return rows
        except Exception as exc:
            logger.warning("list_sessions failed: %s", exc)
            return []

    @hookimpl
    async def save_session(self, session: Any) -> None:
        """Persist the current session's pure data to SQLite.

        Accepts any object that exposes ``session_id``, ``namespace``,
        ``messages``, ``memory``, and ``created_at`` — the engine passes a
        live ``SessionContext`` and this method extracts the pure data.
        """
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return

        try:
            sid = str(session.session_id)
            now = _utcnow()
            metadata_json = json.dumps(dict(session.memory), ensure_ascii=False)

            async with self._db.execute(
                """\
                INSERT INTO sessions (id, created_at, updated_at, namespace, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    sid,
                    session.created_at.timestamp(),
                    now,
                    session.namespace,
                    metadata_json,
                ),
            ):
                pass

            # Replace messages for this session id.
            # TODO Phase 3: replace this full overwrite with incremental append
            # by tracking message_index/turn_id and only inserting rows whose
            # index is greater than the current DB max. This avoids unnecessary
            # I/O and SQLite lock contention for long sessions.
            await self._db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            await self._db.executemany(
                """\
                INSERT INTO messages (session_id, role, content, name, tool_calls)
                VALUES (?, ?, ?, ?, ?)
                """,
                self._iter_message_rows(sid, session.messages),
            )

            # Keep FTS index in sync: remove old entries and re-insert.
            # TODO [Phase 3]: optimize to incremental FTS append by tracking
            # last_indexed_message_id and only inserting rows whose index is
            # greater than the current DB max. This avoids unnecessary I/O and
            # SQLite lock contention for long sessions (>200 turns).
            await self._db.execute("DELETE FROM fts_messages WHERE session_id = ?", (sid,))
            await self._db.executemany(
                """\
                INSERT INTO fts_messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                self._iter_fts_rows(sid, session.messages),
            )
            await self._db.commit()
        except Exception as exc:
            logger.warning("save_session(%s) failed: %s", getattr(session, "session_id", "?"), exc)

    @hookimpl
    async def search_session(self, session_id: str, query: str, limit: int = 10) -> list[MemorySearchResult]:
        """Full-text search within a session using SQLite FTS5.

        Returns BM25-ranked results with highlighted snippets, or an empty
        list if the session is unknown or the query matches nothing.
        """
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return []

        try:
            import uuid

            safe_query = self._sanitize_fts_query(query)
            rows: list[MemorySearchResult] = []
            sql = """\
                SELECT
                    session_id,
                    role,
                    content,
                    snippet(fts_messages, 2, '<mark>', '</mark>', '...', 30) as snippet,
                    bm25(fts_messages) as rank
                FROM fts_messages
                WHERE fts_messages MATCH ?
                  AND session_id = ?
                ORDER BY bm25(fts_messages)
                LIMIT ?
            """
            # NOTE: snippet(fts_messages, 2, ...) 中的 2 是硬编码列索引，
            # 对应 fts_messages 表的第 3 列 (content)，0-indexed:
            #   session_id(0), role(1), content(2)
            # 如果修改表结构，必须同步更新此索引！
            async with self._db.execute(sql, (safe_query, session_id, limit)) as cursor:
                async for row in cursor:
                    rows.append(
                        MemorySearchResult(
                            session_id=row["session_id"],
                            role=row["role"],
                            content=row["content"],
                            snippet=row["snippet"] or "",
                            rank=float(row["rank"]),
                        )
                    )
            return rows
        except Exception as exc:
            logger.warning("search_session(%s, %r) failed: %s", session_id, query, exc)
            return []

    @hookimpl
    async def save_plan(self, plan: WorkflowPlan, session_id: str) -> None:
        """Persist a :class:`WorkflowPlan` to SQLite."""
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return

        try:
            import uuid as _uuid

            steps_json = json.dumps([s.model_dump(mode="json") for s in plan.steps], ensure_ascii=False)
            now = _utcnow()
            async with self._db.execute(
                """\
                INSERT INTO plans (plan_id, session_id, goal, status, progress, steps_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    status = excluded.status,
                    progress = excluded.progress,
                    steps_json = excluded.steps_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(plan.plan_id),
                    session_id,
                    plan.task,
                    plan.status,
                    plan.progress,
                    steps_json,
                    plan.created_at.timestamp(),
                    now,
                ),
            ):
                pass
            await self._db.commit()
        except Exception as exc:
            logger.warning("save_plan(%s) failed: %s", getattr(plan, "plan_id", "?"), exc)

    @hookimpl
    async def load_plan(self, plan_id: str) -> WorkflowPlan | None:
        """Load a persisted :class:`WorkflowPlan` from SQLite."""
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return None

        try:
            import datetime as _datetime

            async with self._db.execute(
                "SELECT plan_id, session_id, goal, status, progress, steps_json, created_at, updated_at FROM plans WHERE plan_id = ?",
                (plan_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None

            steps = []
            for step_data in json.loads(row["steps_json"]):
                steps.append(WorkflowStep.model_validate(step_data))

            return WorkflowPlan(
                plan_id=uuid.UUID(row["plan_id"]),
                task=row["goal"],
                steps=tuple(steps),
                status=row["status"],
                progress=row["progress"],
                created_at=_datetime.datetime.fromtimestamp(row["created_at"], tz=_datetime.timezone.utc),
            )
        except Exception as exc:
            logger.warning("load_plan(%s) failed: %s", plan_id, exc)
            return None

    @hookimpl
    async def list_plans(self, session_id: str) -> list[WorkflowPlan]:
        """List all plans for a given session."""
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return []

        try:
            import datetime as _datetime

            plans: list[WorkflowPlan] = []
            async with self._db.execute(
                "SELECT plan_id, goal, status, progress, steps_json, created_at FROM plans WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ) as cursor:
                async for row in cursor:
                    steps = []
                    for step_data in json.loads(row["steps_json"]):
                        steps.append(WorkflowStep.model_validate(step_data))
                    plans.append(
                        WorkflowPlan(
                            plan_id=uuid.UUID(row["plan_id"]),
                            task=row["goal"],
                            steps=tuple(steps),
                            status=row["status"],
                            progress=row["progress"],
                            created_at=_datetime.datetime.fromtimestamp(row["created_at"], tz=_datetime.timezone.utc),
                        )
                    )
            return plans
        except Exception as exc:
            logger.warning("list_plans(%s) failed: %s", session_id, exc)
            return []

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #
    def _memory_search_spec(self) -> ToolSpec:
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
    def get_tool_specs(self, context: SessionContext) -> tuple[ToolSpec, ...]:
        return (self._memory_search_spec(),)

    @hookimpl
    async def execute_tool(self, context: SessionContext, tool: ToolSpec, arguments: dict[str, object]) -> ToolResult | None:
        if tool.name != "memory_search":
            return None
        query = str(arguments.get("query", ""))
        limit = int(arguments.get("limit", 5))
        if not query:
            return ToolResult(tool_name="memory_search", status=ToolResultStatus.ERROR, error="缺少 'query' 参数。", output={})

        session_id = str(context.session_id)
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

    @hookimpl
    async def delete_session(self, session_id: str) -> bool:
        """Delete a persisted session and all its messages from the store."""
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return False

        try:
            async with self._db.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return False

            await self._db.execute("DELETE FROM fts_messages WHERE session_id = ?", (session_id,))
            await self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            await self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await self._db.commit()
            return True
        except Exception as exc:
            logger.warning("delete_session(%s) failed: %s", session_id, exc)
            return False

    @hookimpl
    async def rename_session(self, session_id: str, title: str) -> SessionData | None:
        """Rename a persisted session by updating its metadata."""
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return None

        try:
            async with self._db.execute("SELECT id, metadata FROM sessions WHERE id = ?", (session_id,)) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None

            metadata = json.loads(row["metadata"] or "{}")
            metadata["title"] = title
            now = _utcnow()
            await self._db.execute(
                "UPDATE sessions SET metadata = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), now, session_id),
            )
            await self._db.commit()

            # Return updated SessionData
            messages = await self._load_messages(session_id)
            import datetime as _datetime
            import uuid as _uuid
            return SessionData(
                session_id=_uuid.UUID(session_id),
                namespace=row["namespace"] if "namespace" in row.keys() else "default",
                messages=tuple(messages),
                memory=metadata,
                created_at=_datetime.datetime.fromtimestamp(
                    _utcnow(), tz=_datetime.timezone.utc
                ),
            )
        except Exception as exc:
            logger.warning("rename_session(%s) failed: %s", session_id, exc)
            return None

    @hookimpl
    async def clear_sessions(self) -> int:
        """Delete all persisted sessions and their messages from the store."""
        if self._db is None:
            await self.initialize()
        if self._db is None:
            return 0

        try:
            await self._db.execute("DELETE FROM fts_messages")
            await self._db.execute("DELETE FROM messages")
            await self._db.execute("DELETE FROM sessions")
            await self._db.commit()
            return 1  # SQLite doesn't return rowcount for DELETE without WHERE
        except Exception as exc:
            logger.warning("clear_sessions failed: %s", exc)
            return 0

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _load_messages(self, session_id: str) -> list[Message]:
        """Load all messages for ``session_id`` and return as Pydantic models."""
        rows: list[Message] = []
        try:
            async with self._db.execute(
                "SELECT role, content, name, tool_calls FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ) as cursor:
                async for row in cursor:
                    tool_calls = json.loads(row["tool_calls"] or "[]")
                    rows.append(
                        Message(
                            role=row["role"],
                            content=row["content"],
                            name=row["name"],
                            tool_calls=tuple(ToolCall(**tc) for tc in tool_calls),
                        )
                    )
        except Exception as exc:
            logger.warning("_load_messages(%s) failed: %s", session_id, exc)
        return rows

    def _iter_message_rows(self, session_id: str, messages: Sequence[Message]) -> Sequence[tuple]:
        """Yield parameter tuples for executemany."""
        result: list[tuple] = []
        for msg in messages:
            tool_calls_json = json.dumps(
                [tc.model_dump(mode="json") for tc in msg.tool_calls], ensure_ascii=False
            )
            result.append((session_id, msg.role, msg.content, msg.name, tool_calls_json))
        return result

    def _iter_fts_rows(self, session_id: str, messages: Sequence[Message]) -> Sequence[tuple]:
        """Yield (session_id, role, content) tuples for FTS5 insertion."""
        result: list[tuple] = []
        for msg in messages:
            result.append((session_id, msg.role, msg.content))
        return result

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize ``query`` for safe use in an FTS5 ``MATCH`` expression.

        FTS5 treats bare words as phrases but interprets special tokens such
        as ``AND``, ``OR``, ``NOT`` and ``*`` as boolean operators /
        wildcards. Because query strings originate from untrusted LLM output,
        we escape the entire input by wrapping it in double quotes and
        escaping any embedded quotes, forcing phrase-match semantics.
        """
        sanitized = query.replace('"', '""')
        return f'"{sanitized}"'


__all__ = ["SQLiteSessionStorePlugin"]
