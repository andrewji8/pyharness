"""SessionContext — task-isolated, reversible agent state.

Reversibility principle
-----------------------
State is **never mutated in place**. Every transition produces a new snapshot
via ``model_copy(update=...)``; ``SessionContext`` is a frozen pydantic model, so
the object graph cannot change after creation. Because each snapshot records its
``branch_id`` and ``parent_id``, you can:

* **roll back** — hold an old snapshot and ask the loop to resume from it;
* **explore branches** — derive divergent futures from one ancestor (the
  spatiotemporal-composability idea from the Cordis paper).

Context isolation
-----------------
A ``ContextVar`` scoped to the running ``asyncio`` task holds *the current*
``SessionContext``. Each concurrent session gets its own context; nothing global
is shared. The recommended way to enter one is :func:`open_session`, which sets
the variable for the duration of the async block.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from datetime import datetime
from typing import Any, AsyncIterator

from pydantic import Field

from pyharness.schema import Frozen, Message, _utcnow

# Task-scoped handle to the "current" session snapshot.
_current_context: ContextVar["SessionContext | None"] = ContextVar(
    "pyharness_current_context", default=None
)

# Track subagent nesting depth to prevent runaway recursion.
_subagent_depth: ContextVar[int] = ContextVar("pyharness_subagent_depth", default=0)


def current_context() -> "SessionContext | None":
    """Return the :class:`SessionContext` bound to the current task, or ``None``."""
    return _current_context.get()


def require_context() -> "SessionContext":
    """Return the current context, raising if the caller is outside any session."""
    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError(
            "No active SessionContext. Call within `open_session(...)`."
        )
    return ctx


class SessionContext(Frozen):
    """Immutable snapshot of one session's state on the event thread.

    ``messages`` and ``memory`` are the durable, schema-driven state. Everything
    else is provenance for reversibility (branching) and observability.
    """

    session_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    branch_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    parent_id: uuid.UUID | None = Field(default=None, description="Branch this snapshot derives from.")
    namespace: str = "default"

    messages: tuple[Message, ...] = Field(default_factory=tuple)
    memory: dict[str, Any] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # -- Reversibility ------------------------------------------------------ #
    def derive(self, *, branch: bool = True, **updates: Any) -> "SessionContext":
        """Produce a new snapshot from this one without mutating it.

        When ``branch=True`` the child gets its own ``branch_id`` and links back
        to ``parent_id``, yielding a reversible, forkable lineage. Pass
        ``branch=False`` for trivial extensions that keep the same branch
        identity (e.g. appending a single message mid-step).
        """
        base_id = self.branch_id
        kwargs = {"updated_at": _utcnow(), **updates}
        if branch:
            kwargs["parent_id"] = self.branch_id
            kwargs["branch_id"] = uuid.uuid4()
        else:
            kwargs.setdefault("branch_id", base_id)
        return self.model_copy(update=kwargs)

    def append_message(self, message: Message) -> "SessionContext":
        """Append one immutable :class:`Message`, returning the new snapshot."""
        return self.derive(messages=(*self.messages, message))

    def with_memory(self, **items: Any) -> "SessionContext":
        """Fold ``items`` into ``memory`` as a new snapshot (reversible)."""
        memory = dict(self.memory)
        memory.update(items)
        return self.derive(memory=memory)

    @property
    def last_message(self) -> "Message | None":
        """Most recent message, if any."""
        return self.messages[-1] if self.messages else None

    def summary(self) -> str:
        """Short human-readable digest for logs/CLI."""
        return (
            f"session={self.session_id.hex[:8]} branch={self.branch_id.hex[:8]} "
            f"msgs={len(self.messages)} mem={len(self.memory)}"
        )


@asynccontextmanager
async def open_session(
    *,
    messages: tuple[Message, ...] = (),
    memory: dict[str, Any] | None = None,
    namespace: str = "default",
    context: "SessionContext | None" = None,
    **meta: Any,
) -> AsyncIterator["SessionContext"]:
    """Async context manager that binds a :class:`SessionContext` to the current
    task.

    With ``context=None`` (default) a **fresh** snapshot is minted from
    ``messages`` / ``memory`` / ``namespace``. Pass ``context=<snapshot>`` to
    continue an existing reversible lineage instead — the same frozen snapshot is
    bound, and any derives during the block (``append_message`` etc.) branch off
    it, never mutating it. On clean exit the ContextVar is reset, so concurrent
    sessions never leak state into one another.

    ::

        async with open_session(memory={"cwd": "/tmp"}) as ctx:
            prompt(ctx)          # reads via require_context()
    """
    token: Token["SessionContext | None"] | None = None
    if context is not None:
        ctx = context
    else:
        ctx = SessionContext(
            messages=messages,
            memory=memory or {},
            namespace=namespace,
            metadata=meta,
        )
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        if token is not None:
            _current_context.reset(token)


def set_current(context: "SessionContext") -> Token["SessionContext | None"]:
    """Explicitly bind ``context`` to the current task. Returns a reset Token."""
    return _current_context.set(context)


def reset_current(token: Token["SessionContext | None"]) -> None:
    """Undo a :func:`set_current` binding (see :func:`contextvars.Token`)."""
    _current_context.reset(token)


__all__ = [
    "SessionContext",
    "current_context",
    "open_session",
    "require_context",
    "reset_current",
    "set_current",
]