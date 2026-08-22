"""FastAPI application exposing PyHarness as a web plugin.

Endpoints
---------
* ``GET  /health``   → ``{status, models}`` (provider negotiation).
* ``POST /chat``     → one turn; pass ``session_id`` to continue a conversation.
* ``WS   /ws/chat``  → streaming: send ``{"text": ...}``, receive ``delta`` and
  ``end`` frames; send ``{"action": "reset"}`` to start a fresh lineage.

State is kept as :class:`~pyharness.context.SessionContext` snapshots in an
in-process :class:`SessionStore`, so multi-turn conversations continue on the
same reversible lineage (``continue_from``).
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from pyharness import Harness, AgentConfig
from pyharness.context import SessionContext
from pyharness.schema import HarnessConfig

logger = logging.getLogger("pyharness.web")

# --------------------------------------------------------------------------- #
# Schemas (schema-driven REST surface)
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    """One user turn. ``session_id`` resumes an existing conversation."""

    text: str = Field(..., min_length=1)
    model: str = Field(default="deepseek-chat")
    system_prompt: str = Field(default="")
    name: str = Field(default="web-agent")
    session_id: str | None = Field(default=None)


class TurnMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    messages: list[TurnMessage]


class SessionStore:
    """Single-process, in-memory map session_id → latest SessionContext."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionContext] = {}

    def put(self, session_id: str, ctx: SessionContext) -> None:
        self._sessions[session_id] = ctx

    def get(self, session_id: str | None) -> SessionContext | None:
        return self._sessions.get(session_id) if session_id else None

    def clear(self) -> None:
        self._sessions.clear()


store = SessionStore()


@lru_cache(maxsize=1)
def get_harness() -> Harness:
    """A cached harness (auto-loads builtin + llm entry-points)."""
    return Harness(config=HarnessConfig(auto_load_entry_points=True))


def _agent(request: ChatRequest) -> AgentConfig:
    return AgentConfig(name=request.name, model=request.model, system_prompt=request.system_prompt)


def _messages(ctx: SessionContext) -> list[TurnMessage]:
    return [TurnMessage(role=m.role.value, content=m.content) for m in ctx.messages]


def _provider_models() -> list[str]:
    """Flatten the fan-in ``get_llm_providers`` hook across all plugins."""
    names: set[str] = set()
    for group in get_harness().bus.pm.hook.get_llm_providers(context=SessionContext()):
        names.update(group)
    return sorted(names)


app = FastAPI(title="PyHarness Web API", version="0.1.0")

# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health() -> dict[str, object]:
    """Liveness + the configured provider model names."""
    return {"status": "ok", "models": _provider_models()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run one user turn (resuming ``session_id`` when given)."""
    harness = get_harness()
    agent = _agent(request)
    prior = store.get(request.session_id)
    ctx = await harness.run_session(agent, initial_text=request.text, continue_from=prior)
    store.put(str(ctx.session_id), ctx)
    return ChatResponse(
        session_id=str(ctx.session_id),
        reply=ctx.last_message.content if ctx.last_message else "",
        messages=_messages(ctx),
    )


# --------------------------------------------------------------------------- #
# WebSocket (streaming)
# --------------------------------------------------------------------------- #
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    """Stream one or more turns over a persistent socket.

    Client frames:
        {"text": "hi", "model": "demo", "system_prompt": "", "name": "web-agent"}
        {"action": "reset"}        # start a fresh lineage
    Server frames:
        {"type": "delta", "delta": "..."}
        {"type": "end", "session_id": "...", "reply": "..."}
    """
    await ws.accept()
    harness = get_harness()
    state: SessionContext | None = None
    try:
        while True:
            raw = await ws.receive_json()
            if raw.get("action") == "reset":
                state = None
                await ws.send_json({"type": "reset_ok"})
                continue
            text = raw.get("text")
            if not text:
                continue
            agent = AgentConfig(
                name=raw.get("name", "web-agent"),
                model=raw.get("model", "deepseek-chat"),
                system_prompt=raw.get("system_prompt", ""),
            )
            async for chunk in harness.stream_session(agent, initial_text=text, continue_from=state):
                if chunk.delta:
                    await ws.send_json({"type": "delta", "delta": chunk.delta})
            state = harness.last_context
            if state is not None:
                store.put(str(state.session_id), state)
                await ws.send_json(
                    {
                        "type": "end",
                        "session_id": str(state.session_id),
                        "reply": state.last_message.content if state.last_message else "",
                    }
                )
            else:
                await ws.send_json({"type": "end", "session_id": None, "reply": ""})
    except WebSocketDisconnect:
        logger.info("ws /ws/chat disconnected")


__all__ = ["ChatRequest", "ChatResponse", "SessionStore", "app", "store"]