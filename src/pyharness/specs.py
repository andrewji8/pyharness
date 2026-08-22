"""AgentHooks — the pluggy Hookspec that defines PyHarness's *entire* contract.

Everything is a plugin: LLM providers, tool executors, CLI/Web UI, persistence
and even the core's own lifecycle are reached exclusively through these hooks.
The thin core never imports a concrete plugin; it only calls ``self.hooks.<name>``.

Design notes
------------
* Synchronous spec methods that merely **observe** (lifecycle, events, tool
  listing) let plugins use fast, synchronous ``@hookimpl`` bodies.
* Methods that **perform work** (llm, tools, agent step) are ``async``; the
  engine awaits the returned coroutines. pluggy ships coroutines back to the
  caller unchanged, so both styles interoperate cleanly.
* Provider hooks are **fan-in**: pluggy returns one value/iterator per plugin;
  the engine awaits them in registration order and consumes the first non-``None``.
  (pluggy's ``firstresult`` does **not** fall through on ``None`` for coroutine
  bodies, so provider fallback intentionally lives in the engine, not in pluggy.)

HookspecMarker is instantiated once; plugins use the same project name to
declare their ``@hookimpl`` markers (see ``plugins/builtin.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from pluggy import HookspecMarker

from pyharness.schema import AgentConfig, Chunk, Event, MCPServerConfig, Message, MemorySearchResult, SearchResult, SessionData, SubagentResult, SubagentSpec, ToolCall, ToolResult, ToolSpec, WorkflowPlan, WorkflowStep
from pyharness.context import SessionContext

if TYPE_CHECKING:
    from pyharness.core import Harness

hookspec = HookspecMarker("pyharness")


class AgentHooks:
    """Declarative hook contract. Concrete ``@hookimpl`` live inside plugins."""

    # -- Engine lifecycle --------------------------------------------------- #
    @hookspec
    def harness_initialized(self, harness: "Harness") -> None:
        """Emitted once after entry-point plugins are loaded. Good for wiring."""

    @hookspec
    def harness_shutdown(self, harness: "Harness") -> None:
        """Emitted on graceful shutdown. Use for resource cleanup."""

    # -- Session lifecycle -------------------------------------------------- #
    @hookspec
    def session_started(self, context: SessionContext, agent: AgentConfig | None = None) -> None:
        """A new isolated context was opened inside a task."""

    @hookspec
    def session_finished(self, context: SessionContext) -> None:
        """The context is being closed (best-effort final state)."""

    @hookspec
    async def load_session(self, session_id: str) -> SessionData | None:
        """Attempt to restore a previously persisted session by ``session_id``.

        Return a :class:`SessionData` DTO (pure data, no runtime state) when the
        store has history for this id, or ``None`` if the session is unknown.
        The engine will wrap the returned DTO in a fresh
        :class:`~pyharness.context.SessionContext` so that ``ContextVar``
        bindings and reversibility lineage are always freshly minted."""

    @hookspec
    async def save_session(self, session: SessionContext) -> None:
        """Persist the current ``session`` snapshot to durable storage.

        Called at key lifecycle points (end of turn, session finish). Store
        plugins should extract pure data from the ``SessionContext`` and persist
        that; they must not attempt to serialize runtime state like
        ``ContextVar`` bindings. Failures must not crash the agent loop."""

    @hookspec
    async def search_session(self, session_id: str, query: str, limit: int = 10) -> list[MemorySearchResult]:
        """Full-text search within a session's message history.

        Return a list of :class:`MemorySearchResult` ordered by BM25 relevance
        (best matches first). Store plugins with FTS or similar indexes should
        implement this hook; if no plugin answers, the engine treats it as an
        empty result set."""

    @hookspec
    async def spawn_subagent(self, spec: SubagentSpec, parent_tools: list[ToolSpec], parent_config: AgentConfig) -> SubagentResult:
        """Spawn a worker subagent to execute ``spec.task`` independently.

        The engine supplies the parent's tool list and config so the worker
        can inherit tooling and model settings. Implementations must respect
        ``spec.timeout`` and ``spec.max_turns``. Returning a non-``None``
        result short-circuits fan-out: the first plugin that answers wins."""

    @hookspec
    async def subagent_finished(self, result: SubagentResult) -> None:
        """Emitted when a subagent run completes (success, timeout, or error)."""

    # -- Model providers (async; engine-side fallback over all impls) ------- #
    @hookspec
    async def llm_complete(self, context: SessionContext, request: Any) -> Any:
        """Return an ``LLMResponse``. Async ``@hookimpl`` coroutines are all
        run; the engine awaits them in registration order and consumes the first
        non-``None`` value (pluggy's ``firstresult`` does **not** fall through on
        ``None`` for coroutine bodies, so fallback lives in the engine)."""

    @hookspec
    def get_llm_providers(self, context: SessionContext) -> tuple[str, ...]:
        """Contribute provider/model names for capability negotiation."""

    # -- Tools --------------------------------------------------------------- #
    @hookspec
    def get_tool_specs(self, context: SessionContext) -> tuple[ToolSpec, ...]:
        """Return the declarative specs of tools this plugin provides."""

    @hookspec
    async def execute_tool(self, context: SessionContext, tool: ToolSpec, arguments: dict[str, Any]) -> ToolResult:
        """Execute ``tool`` with ``arguments``. Engine awaits all impls and
        takes the first non-``None`` ``ToolResult`` (owner returns a result,
        non-owners return ``None``)."""

    # -- The agent loop ------------------------------------------------------ #
    @hookspec
    async def agent_next(self, context: SessionContext, agent: AgentConfig) -> Any:
        """Decide the next agent action. Return None to defer to other plugins;
        otherwise return an ``LLMResponse``-like object. Engine consumes the
        first non-``None`` value across all awaited impls."""

    @hookspec
    async def llm_stream(self, context: SessionContext, request: Any) -> AsyncIterator[Any]:
        """Optionally stream deltas for ``request`` as an async generator of
        :class:`~pyharness.schema.LLMStreamChunk`. Only the first provider that
        yields is consumed by the engine. Implementers *must* be async
        generators (``async def ... yield``)."""

    # -- Events (broadcast sink to the whole plugin set) --------------------- #
    @hookspec
    def observe(self, context: SessionContext, event: Event) -> None:
        """Every event emitted for a session is fanned out to all plugins.
        Observers (logging, metrics, persistence) implement this hook."""

    @hookspec
    async def list_sessions(self, namespace: str = "default", limit: int = 50, offset: int = 0) -> list[SessionData]:
        """List persisted sessions, ordered by most recent first.

        UI plugins use this to render the session sidebar."""

    @hookspec
    async def check_shutdown(self, context: SessionContext, agent: AgentConfig) -> bool:
        """Return True to end the loop. Plugins can layer stopping conditions
        (max tokens, user ctrl-c, safe-mode). Default/None: keep running."""

    @hookspec
    async def pre_tool_execution(self, tool_call: ToolCall) -> ToolResult | None:
        """Intercept a tool call before it is executed. Return a ToolResult to
        short-circuit execution (the engine will use this result instead of
        calling the actual tool), or None to allow execution to proceed."""

    @hookspec
    async def ask_user_confirmation(self, prompt: str, metadata: dict) -> bool | None:
        """Request human confirmation for a sensitive action. Return True to
        allow, False to reject, or None to defer to other plugins. UI plugins
        (CLI, Web) implement this hook to present a confirmation dialog."""

    @hookspec
    async def build_request(self, messages: list[Message]) -> list[Message]:
        """Intercept and optionally transform the message list before it is sent
        to the LLM provider. Return a new list of :class:`Message` objects.
        Plugins such as context compaction use this to inject summaries or
        prune history."""

    # -- Workflow lifecycle --------------------------------------------------- #
    @hookspec
    async def on_plan_created(self, plan: WorkflowPlan) -> None:
        """Emitted when a new :class:`WorkflowPlan` has been generated.

        UI plugins can use this to render the To-Do List; persistence plugins
        can snapshot the plan for later inspection."""
        pass

    @hookspec
    async def on_step_update(self, plan_id: str, step: WorkflowStep) -> None:
        """Emitted when a single :class:`WorkflowStep` changes state.

        UI plugins can update progress bars; Guard plugins can intercept
        sensitive steps before they run."""
        pass

    @hookspec
    async def on_plan_completed(self, plan: WorkflowPlan) -> None:
        """Emitted when every step in the plan has reached a terminal state.

        Persistence plugins can write the final plan record; reporting plugins
        can emit a summary."""
        pass

    @hookspec
    async def on_plan_update(self, plan: WorkflowPlan) -> None:
        """Emitted when a plan is dynamically modified via ``update_plan``.

        UI plugins can refresh the plan view; persistence plugins can write
        the updated plan record."""
        pass

    @hookspec
    async def save_plan(self, plan: WorkflowPlan, session_id: str) -> None:
        """Persist a :class:`WorkflowPlan` to durable storage.

        Called after plan creation, step updates, and plan completion/cancellation.
        Implementations should serialize the plan and associate it with ``session_id``."""
        pass

    @hookspec
    async def load_plan(self, plan_id: str) -> WorkflowPlan | None:
        """Load a previously persisted :class:`WorkflowPlan` by its ``plan_id``.

        Return the plan if found, or ``None`` if the plan does not exist."""
        pass

    @hookspec
    async def list_plans(self, session_id: str) -> list[WorkflowPlan]:
        """List all plans associated with a given ``session_id``.

        Return a list of :class:`WorkflowPlan` objects, ordered by creation time."""
        pass

    # -- RAG / Knowledge --------------------------------------------------- #
    @hookspec(firstresult=True)
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert a batch of texts into embedding vectors.

        Return a list of float lists (one per input text). The first non-``None``
        result wins; if no plugin implements this, the engine falls back to a
        deterministic dummy embedding."""
        pass

    @hookspec(firstresult=True)
    async def embed_query(self, query: str) -> list[float]:
        """Convert a single query string into an embedding vector.

        Return a float list. The first non-``None`` result wins."""
        pass

    @hookspec
    async def store_chunks(self, chunks: list[Chunk]) -> None:
        """Persist a list of :class:`Chunk` objects (with embeddings) to the
        vector store."""
        pass

    @hookspec
    async def search_similar(self, query_vector: list[float], top_k: int = 5, filter: dict | None = None) -> list[SearchResult]:
        """Perform vector similarity search.

        Return the top-k most similar chunks, ordered by descending score."""
        pass

    @hookspec
    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks originating from ``source`` (e.g. a file path).

        Return the number of chunks removed."""
        pass

    @hookspec
    async def get_store_stats(self) -> dict:
        """Return vector store statistics (total chunks, vector dimension, etc.)."""
        pass

    # -- MCP (Model Context Protocol) --------------------------------------- #
    @hookspec
    async def mcp_connect(self, config: MCPServerConfig) -> bool:
        """Connect to an MCP Server.

        Return ``True`` on success, ``False`` on failure."""
        pass

    @hookspec
    async def mcp_disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP Server by name."""
        pass

    @hookspec
    async def mcp_list_tools(self, server_name: str) -> list[dict]:
        """List tools exposed by a connected MCP Server.

        Return a list of tool descriptors (name, description, inputSchema)."""
        pass

    @hookspec
    async def mcp_call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """Call a tool on a connected MCP Server.

        Return the raw JSON-RPC result dict."""
        pass

    @hookspec
    async def mcp_list_servers(self) -> list[dict]:
        """List all configured/connected MCP Servers and their status."""
        pass


__all__ = ["AgentHooks", "hookspec"]