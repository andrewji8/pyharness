"""Integration tests for the SQLite session store (Phase 2 memory).

Verifies the full lifecycle:

1. Run an agent session with the store plugin active.
2. Verify the session is persisted to SQLite.
3. Resume from the persisted ``session_id`` in a fresh harness.
4. Verify the resumed context contains the original messages.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from pyharness import Harness
from pyharness.context import SessionContext
from pyharness.core import _settle
from pyharness.plugins.llm import entry as llm
from pyharness.plugins.session_store import SQLiteSessionStorePlugin
from pyharness.schema import (
    AgentConfig,
    HarnessConfig,
    LLMResponse,
    MemorySearchResult,
    Message,
    Role,
    SessionData,
)


def _harness_with_store(db_path: str, *, auto_load: bool = False) -> tuple[Harness, SQLiteSessionStorePlugin]:
    """Build a Harness wired with the LLM plugin and a SQLite session store."""
    h = Harness(config=HarnessConfig(auto_load_entry_points=auto_load))
    llm.clear()
    llm.use_dummy(models=("dummy",), plan=[LLMResponse(model="dummy", content="hi there")])
    h.register_plugin(llm)  # LLM plugin must be explicitly registered
    store = SQLiteSessionStorePlugin(db_path=db_path)
    h.register_plugin(store)
    h.initialize()
    return h, store


@pytest.fixture()
def tmp_db(tmp_path: str) -> str:
    """Provide an isolated SQLite file path for each test."""
    return str(tmp_path / "pyharness_test.db")


async def test_session_is_persisted_and_resumable(tmp_path: str) -> None:
    db_path = str(tmp_path / "memory.db")
    h, store = _harness_with_store(db_path)

    try:
        # --- Turn 1: fresh session, save to SQLite ---
        final = await h.run_session(
            AgentConfig(name="mem-agent", model="dummy"),
            initial_text="hello",
        )
        session_id = str(final.session_id)
        assert session_id

        # Verify DB file created and session row exists.
        assert os.path.exists(db_path)
        rows = await store._db.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
        )
        count = (await rows.fetchone())[0]
        assert count == 1

        # --- Turn 2: resume from store in a FRESH harness ---
        h2, store2 = _harness_with_store(db_path)
        resumed = await h2.run_session(
            AgentConfig(name="mem-agent", model="dummy"),
            resume_session_id=session_id,
            initial_text="second turn",
        )

        # The resumed context should carry the full history.
        roles = [m.role for m in resumed.messages]
        assert Role.USER in roles
        assert Role.ASSISTANT in roles
        assert len(resumed.messages) >= 3  # hello + hi there + second turn + next reply
        await store2.teardown()
    finally:
        await store.teardown()


async def test_load_missing_session_returns_none(tmp_path: str) -> None:
    db_path = str(tmp_path / "empty.db")
    h, store = _harness_with_store(db_path)

    try:
        results = await _settle(h.bus.pm.hook.load_session(session_id="00000000-0000-0000-0000-000000000000"))
        assert results == [None]
    finally:
        await store.teardown()


async def test_lazy_initialization_on_save(tmp_path: str) -> None:
    """Verify save_session auto-initializes the DB even without explicit initialize()."""
    db_path = str(tmp_path / "lazy.db")
    store = SQLiteSessionStorePlugin(db_path=db_path)
    ctx = SessionContext()
    await store.save_session(ctx)
    # DB file SHOULD have been created because save_session now lazy-initializes.
    assert os.path.exists(db_path)
    await store.teardown()


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="pyharness_memory_"))
    try:
        asyncio.run(test_session_is_persisted_and_resumable(tmp))
        print("PASS: test_session_is_persisted_and_resumable")
        asyncio.run(test_load_missing_session_returns_none(tmp))
        print("PASS: test_load_missing_session_returns_none")
        asyncio.run(test_lazy_initialization_on_save(tmp))
        print("PASS: test_lazy_initialization_on_save")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def test_fts_search_returns_ranked_results(tmp_path: str) -> None:
    db_path = str(tmp_path / "search.db")
    h, store = _harness_with_store(db_path)

    try:
        final = await h.run_session(
            AgentConfig(name="search-agent", model="dummy"),
            initial_text="Python is a great programming language",
        )
        session_id = str(final.session_id)

        # Search for a keyword that appeared in the conversation.
        results = await store.search_session(session_id=session_id, query="Python", limit=10)
        assert len(results) >= 1
        top = results[0]
        assert isinstance(top, MemorySearchResult)
        assert "Python" in top.content or "Python" in top.snippet
        assert top.rank <= 0.0  # BM25: lower is better
        assert top.session_id == str(final.session_id)
    finally:
        await store.teardown()


async def test_fts_search_scoped_to_session(tmp_path: str) -> None:
    db_path = str(tmp_path / "scoped.db")
    h1, store1 = _harness_with_store(db_path)
    h2, store2 = _harness_with_store(db_path)

    try:
        # Session A: talks about Python
        ctx_a = await h1.run_session(
            AgentConfig(name="a", model="dummy"),
            initial_text="I love Python",
        )
        # Session B: talks about Rust
        ctx_b = await h2.run_session(
            AgentConfig(name="b", model="dummy"),
            initial_text="I love Rust",
        )

        # Search session A for "Python" — should NOT return session B's messages.
        results_a = await store1.search_session(session_id=str(ctx_a.session_id), query="Python", limit=10)
        assert all(r.session_id == str(ctx_a.session_id) for r in results_a)

        # Search session B for "Rust" — should NOT return session A's messages.
        results_b = await store2.search_session(session_id=str(ctx_b.session_id), query="Rust", limit=10)
        assert all(r.session_id == str(ctx_b.session_id) for r in results_b)
    finally:
        await store1.teardown()
        await store2.teardown()


async def test_fts_search_sanitizes_special_characters(tmp_path: str) -> None:
    """Verify FTS5 special characters in queries are escaped safely."""
    db_path = str(tmp_path / "sanitize.db")
    store = SQLiteSessionStorePlugin(db_path=db_path)
    try:
        await store.initialize()

        # These queries contain FTS5 special syntax that should be neutralized.
        dangerous_queries = [
            'python AND "delete table"',
            "error OR warning",
            "NOT important",
            "test*",
            'quote "inside" query',
        ]
        for q in dangerous_queries:
            # Should not raise; sanitized query should be a quoted phrase.
            results = await store.search_session(
                session_id="00000000-0000-0000-0000-000000000000",
                query=q,
                limit=5,
            )
            assert isinstance(results, list)
    finally:
        await store.teardown()


async def test_memory_search_tool_execution(tmp_path: str) -> None:
    db_path = str(tmp_path / "tool.db")
    h, store = _harness_with_store(db_path)

    try:
        final = await h.run_session(
            AgentConfig(name="tool-agent", model="dummy"),
            initial_text="Remember: buy milk and eggs",
        )
        session_id = str(final.session_id)

        # Resolve the memory_search tool spec.
        specs: dict[str, Any] = {}
        for plugin_specs in h.bus.pm.hook.get_tool_specs(context=final):
            specs.update({s.name: s for s in plugin_specs})
        tool = specs.get("memory_search")
        assert tool is not None

        result = await store.execute_tool(final, tool, {"query": "milk", "limit": 5})
        assert result.status.value == "ok"
        assert result.output["count"] >= 1
        assert "milk" in str(result.output["results"][0]["content"])
    finally:
        await store.teardown()
