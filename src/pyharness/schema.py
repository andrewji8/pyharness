"""Core schema layer.

Every piece of data that crosses the plugin boundary is a strict, immutable
``pydantic`` model. This module is the **single source of truth** for the shapes
the engine and plugins agree on (Schema-Driven principle). Defining these here
lets the thin core describe tools, events, configs and LLM traffic without
hard-coding any provider or executor logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp used for all snapshot timestamps."""
    return datetime.now(timezone.utc)


class Frozen(BaseModel):
    """Base for all schema models: immutable → reversible via ``model_copy``."""

    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- #
# Events — the observations that flow over the EventBus.
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    """Canonical event names. Plugins may also emit free-form event types."""

    USER_MESSAGE = "user.message"
    ASSISTANT_FINISHED = "assistant.finished"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    STEP = "step.completed"
    SESSION_STARTED = "session.started"
    SESSION_FINISHED = "session.finished"


class Event(Frozen):
    """An immutable observation emitted into the EventBus.

    ``session_id`` (not the ``SessionContext`` object itself) is tracked, so the
    event layer never couples to a live context — contexts live only in
    ``contextvars`` and are addressed by id.
    """

    type: str = Field(description="Event type; use EventType for canonical ones.")
    session_id: uuid.UUID = Field(description="Owning session id.")
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class ToolArg(Frozen):
    """A single tool parameter."""

    name: str
    type: str = Field(description="JSON-schema-ish type: string | integer | number | boolean | array | object")
    description: str = ""
    required: bool = False
    enum: tuple[str, ...] = Field(default_factory=tuple)


class ToolSpec(Frozen):
    """Declarative description of a callable tool (the tool executor is a plugin)."""

    name: str
    description: str = ""
    parameters: tuple[ToolArg, ...] = Field(default_factory=tuple)
    timeout_seconds: float = Field(default=10.0, ge=0.1)
    # Secrets (api keys etc.) are stripped before the tool payload is emitted.
    secret: bool = False

    def arg_names(self) -> set[str]:
        """Names of declared parameters, for cheap validation."""
        return {p.name for p in self.parameters}


class ToolResultStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


class ToolResult(Frozen):
    """Outcome of executing a tool."""

    tool_name: str
    status: ToolResultStatus = ToolResultStatus.OK
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None)
    duration_seconds: float = Field(default=0.0, ge=0.0)


# --------------------------------------------------------------------------- #
# LLM traffic (transport-agnostic; providers are plugins)
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(Frozen):
    """One turn in a conversation slice."""

    role: Role
    content: str
    name: str | None = Field(default=None, description="Tool/role name, when relevant.")
    tool_calls: tuple["ToolCall", ...] = Field(default_factory=tuple)
    tool_call_id: str | None = Field(default=None, description="Tool call ID for tool messages.")


class ToolCall(Frozen):
    """A request for the tool executor to run a tool."""

    id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMRequest(Frozen):
    """Input to a provider. Providers (plugins) translate this to their API."""

    model: str
    messages: tuple[Message, ...] = Field(default_factory=tuple)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    tools: list[dict[str, Any]] = Field(default_factory=list, description="OpenAI-format tool definitions for function calling.")


class LLMResponse(Frozen):
    """Output of a provider, normalized so the engine stays provider-agnostic."""

    model: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = Field(default_factory=tuple)
    finish_reason: Literal["stop", "tool_calls", "length"] = "stop"
    usage: dict[str, int] = Field(default_factory=dict)


class LLMStreamChunk(Frozen):
    """One delta slice of a streaming response."""

    delta: str = ""
    tool_calls: tuple[ToolCall, ...] = Field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Multi-agent / subagent
# --------------------------------------------------------------------------- #
class SubagentSpec(Frozen):
    """Declarative specification for spawning a worker subagent."""

    name: str = Field(description="子 Agent 的名称标识")
    task: str = Field(description="分配给子 Agent 的具体任务描述")
    model: str = Field(default="default", description="子 Agent 使用的模型名称")
    system_prompt: str | None = Field(default=None, description="子 Agent 的系统提示词（可选）")
    allowed_tools: list[str] | None = Field(default=None, description="允许使用的工具白名单（None 表示继承全部）")
    max_turns: int = Field(default=5, ge=1, le=20, description="子 Agent 最大循环轮次")
    timeout: float = Field(default=240.0, ge=10, description="执行超时（秒）")


class SubagentResult(Frozen):
    """Outcome of a completed subagent run."""

    spec: SubagentSpec
    status: Literal["ok", "timeout", "error", "cancelled"] = "ok"
    output: str | None = Field(default=None, description="子 Agent 的最终文本输出（可能为空，例如超时时）")
    error: str | None = Field(default=None, description="错误信息（如有）")
    duration_seconds: float = Field(default=0.0, ge=0.0)
    session_id: str = Field(default="", description="子 Agent 会话 ID（字符串）")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class PlanAction(str, Enum):
    """Allowed mutations for the ``update_plan`` tool."""

    ADD_STEP = "add_step"
    SKIP_STEP = "skip_step"
    UPDATE_STEP = "update_step"
    CANCEL_PLAN = "cancel_plan"


class UpdatePlanInput(Frozen):
    """Input schema for the ``update_plan`` tool."""

    plan_id: str = Field(description="要修改的计划 ID")
    action: PlanAction = Field(description="操作类型")
    step_id: str | None = Field(default=None, description="目标步骤 ID（cancel_plan 时不需要）")
    new_title: str | None = Field(default=None, description="新标题（add_step / update_step 时使用）")
    new_description: str | None = Field(default=None, description="新描述（add_step / update_step 时使用）")
    insert_after: str | None = Field(default=None, description="在此步骤之后插入（add_step 时使用）")
    reason: str | None = Field(default=None, description="修改原因（记录到日志）")


class WorkflowStep(Frozen):
    """One step in a structured execution plan."""

    id: str = Field(description="Unique step identifier within the plan.")
    title: str = Field(description="Short display title for the step.")
    description: str = Field(description="What to do in this step.")
    depends_on: list[str] = Field(default_factory=list, description="Step IDs this step depends on.")
    status: StepStatus = Field(default=StepStatus.PENDING)
    result: str | None = Field(default=None)
    error: str | None = Field(default=None)
    max_retries: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    started_at: datetime | None = Field(default=None)
    finished_at: datetime | None = Field(default=None)
    use_subagent: bool = Field(default=False, description="Whether to spawn a subagent for this step.")
    subagent_max_turns: int = Field(default=5, description="Subagent max turns when use_subagent=True.")
    subagent_timeout: float = Field(default=120.0, description="Subagent timeout seconds when use_subagent=True.")
    duration_seconds: float | None = Field(default=None, description="Execution duration in seconds.")


class WorkflowPlan(Frozen):
    """A structured execution plan generated from a task description."""

    plan_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique plan identifier.")
    task: str = Field(description="The original task the plan was generated for.")
    steps: tuple[WorkflowStep, ...] = Field(default_factory=tuple)
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    final_result: str | None = Field(default=None)
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall completion progress (0.0 to 1.0).")
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Session persistence (pure DTO — no runtime state)
# --------------------------------------------------------------------------- #
class SessionData(Frozen):
    """Pure data transfer object for session persistence.

    This DTO contains only serializable session state. It is intentionally
    decoupled from :class:`SessionContext` (which carries runtime state like
    ``ContextVar`` bindings and reversibility lineage). The engine is
    responsible for reconstructing a fresh ``SessionContext`` from a
    ``SessionData`` snapshot after loading it from storage.
    """

    session_id: uuid.UUID
    namespace: str = "default"
    messages: tuple[Message, ...] = Field(default_factory=tuple)
    memory: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    # Precomputed by lightweight list endpoints (COUNT query) so listings never
    # need to load full message bodies. 0 when unknown / not computed.
    message_count: int = 0


class MemorySearchResult(Frozen):
    """One hit from a full-text search over session messages."""

    session_id: str = Field(description="Session id as a string, matching the SQLite TEXT primary key.")
    role: str
    content: str
    snippet: str = Field(default="", description="Highlighted excerpt from the matched content.")
    rank: float = Field(default=0.0, description="BM25 relevance score (lower is better).")


# --------------------------------------------------------------------------- #
# RAG — Retrieval Augmented Generation
# --------------------------------------------------------------------------- #
class Chunk(Frozen):
    """A text chunk with optional embedding vector."""

    chunk_id: str = Field(description="Unique chunk identifier.")
    content: str = Field(description="The chunk text content.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata such as source_file, line_start, line_end.")
    embedding: list[float] | None = Field(default=None, description="Vector embedding (populated by EmbeddingProvider).")


class SearchResult(Frozen):
    """One hit from a vector similarity search."""

    chunk_id: str
    content: str
    score: float = Field(description="Similarity score (0~1, higher is better).")
    metadata: dict[str, Any] = Field(default_factory=dict)


class HybridSearchResult(Frozen):
    """One hit from a hybrid (FTS5 + vector) search."""

    chunk_id: str
    content: str
    fts_score: float = Field(default=0.0, description="BM25 keyword relevance score.")
    vector_score: float = Field(default=0.0, description="Cosine similarity score.")
    hybrid_score: float = Field(default=0.0, description="Fused relevance score.")
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# RAG Tool Inputs
# --------------------------------------------------------------------------- #
class KnowledgeSearchInput(BaseModel):
    """Input schema for the ``knowledge_search`` tool."""

    query: str = Field(description="The question or keyword to search for.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return.")
    source_filter: str | None = Field(default=None, description="Filter by source file path.")


class IngestDirectoryInput(BaseModel):
    """Input schema for the ``ingest_directory`` tool."""

    path: str = Field(description="Directory path to ingest (relative to workspace).")
    patterns: list[str] = Field(default_factory=lambda: ["*.py", "*.md", "*.txt"], description="File glob patterns to include.")
    chunk_size: int = Field(default=1000, description="Chunk size in characters.")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in characters.")


class HybridSearchInput(BaseModel):
    """Input schema for the ``hybrid_search`` tool."""

    query: str = Field(description="The search query.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return.")
    fts_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for FTS5 keyword scores.")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for vector semantic scores.")


# --------------------------------------------------------------------------- #
# MCP — Model Context Protocol
# --------------------------------------------------------------------------- #
class MCPServerConfig(Frozen):
    """MCP Server connection configuration."""

    name: str = Field(description="Server identifier.")
    command: str | None = Field(default=None, description="stdio transport: executable command.")
    args: list[str] = Field(default_factory=list, description="stdio transport: command arguments.")
    env: dict[str, str] = Field(default_factory=dict, description="stdio transport: environment variables.")
    url: str | None = Field(default=None, description="SSE transport: server URL.")
    transport: Literal["stdio", "sse"] = Field(default="stdio", description="Transport type.")
    enabled: bool = Field(default=True, description="Whether to auto-connect on startup.")


class MCPToolMapping(Frozen):
    """Mapping between an MCP tool and a PyHarness ToolSpec."""

    server_name: str = Field(description="Source MCP Server name.")
    tool_name: str = Field(description="Original tool name on the server.")
    harness_name: str = Field(description="PyHarness tool name (prefixed to avoid collisions).")
    description: str = Field(default="", description="Human-readable description.")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for tool arguments.")


class MCPToolResult(Frozen):
    """Result of calling an MCP tool."""

    content: list[dict[str, Any]] = Field(default_factory=list, description="MCP content array.")
    is_error: bool = Field(default=False, description="Whether the call resulted in an error.")
    error_message: str | None = Field(default=None, description="Error message if any.")

    def to_text(self) -> str:
        """Convert the MCP content array to plain text."""
        parts: list[str] = []
        for item in self.content:
            item_type = item.get("type")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type == "image":
                parts.append(f"[Image: {item.get('mimeType', 'unknown')}]")
        return "\n".join(parts) if parts else "(empty response)"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class AgentConfig(Frozen):
    """Static agent description; treated as immutable configuration."""

    name: str
    model: str = "default"
    system_prompt: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_steps: int = Field(default=32, ge=1)
    tools_enabled: bool = True


class HarnessConfig(Frozen):
    """Engine-level wiring, resolved at startup from entry-point loading."""

    auto_load_entry_points: bool = True
    plugin_group: str = "pyharness.plugins"
    # Seed context implicitly namespaces per Harness instance.
    namespace: str = "default"


# Re-export commonly-used aliases for a tidy public surface.
__all__ = [
    "AgentConfig",
    "Event",
    "EventType",
    "Frozen",
    "HarnessConfig",
    "LLMRequest",
    "LLMResponse",
    "LLMStreamChunk",
    "MemorySearchResult",
    "Message",
    "PlanAction",
    "Role",
    "SessionData",
    "StepStatus",
    "SubagentResult",
    "SubagentSpec",
    "ToolArg",
    "ToolCall",
    "ToolResult",
    "ToolResultStatus",
    "ToolSpec",
    "UpdatePlanInput",
    "WorkflowPlan",
    "WorkflowStep",
    "Chunk",
    "SearchResult",
    "HybridSearchResult",
    "KnowledgeSearchInput",
    "IngestDirectoryInput",
    "HybridSearchInput",
    "MCPServerConfig",
    "MCPToolMapping",
    "MCPToolResult",
    "_utcnow",
]