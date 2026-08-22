"""PyHarness — Everything is a Plugin.

A thin-core agent harness. The engine owns only the Event Bus (pluggy),
Context Management (contextvars) and Plugin Loading; every domain capability
(LLM providers, tool executors, CLIs, Web UIs) arrives as a plugin via
``@hookimpl``. State is schema-driven (pydantic v2) and reversible
(``model_copy(update=...)`` snapshots).
"""

from pyharness.context import (
    SessionContext,
    current_context,
    open_session,
    require_context,
)
from pyharness.core import EventBus, Harness
from pyharness.schema import (
    AgentConfig,
    Event,
    EventType,
    HarnessConfig,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    Role,
    StepStatus,
    ToolArg,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
    WorkflowPlan,
    WorkflowStep,
)
from pyharness.specs import AgentHooks

__version__ = "0.6.2"

__all__ = [
    "AgentConfig",
    "AgentHooks",
    "Event",
    "EventBus",
    "EventType",
    "Harness",
    "HarnessConfig",
    "LLMRequest",
    "LLMResponse",
    "LLMStreamChunk",
    "Message",
    "Role",
    "SessionContext",
    "StepStatus",
    "ToolArg",
    "ToolCall",
    "ToolResult",
    "ToolResultStatus",
    "ToolSpec",
    "WorkflowPlan",
    "WorkflowStep",
    "current_context",
    "open_session",
    "require_context",
]