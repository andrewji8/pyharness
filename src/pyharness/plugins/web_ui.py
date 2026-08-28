"""FastAPI Web UI 插件：REST API + WebSocket 实时推送。

设计
----
* ``WebUIPlugin`` 注册到 Harness 后，持有 ``self.harness`` 引用，
  通过 ``bus.aemit`` / ``bus.on`` 与引擎事件总线交互。
* ``WebSocketObserverPlugin`` 实现 ``observe`` hook，
  将引擎事件转发给所有已连接的 WebSocket 客户端。
* 前端通过 WebSocket 发送 ``user_message``，
  后端用 ``stream_session`` 迭代 LLM 流式分片并实时推送。
* 会话状态通过 ``save_session`` / ``load_session`` / ``list_sessions``
  持久化到 SQLite（由 ``SQLiteSessionStorePlugin`` 提供）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any

import aiofiles
import aiosqlite
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pyharness.core import PluginCoreProtectedError, PluginError, PluginNotFoundError

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import AgentConfig, Event, SessionData
from pyharness.specs import AgentHooks
from pyharness.plugins.ui_websocket import WebSocketObserverPlugin

hookimpl = HookimplMarker("pyharness")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request/response DTOs
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    content: str
    session_id: str | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    status: str


class SearchRequest(BaseModel):
    query: str
    session_id: str
    limit: int = 5


class PluginLoadRequest(BaseModel):
    path: str


class PluginReloadRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# WebUIPlugin
# ---------------------------------------------------------------------------
class WebUIPlugin:
    """FastAPI Web UI 插件。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 3080) -> None:
        self.host = host
        self.port = port
        self.app = FastAPI(title="PyHarness Web UI")
        self.harness: Any = None
        self.ws_clients: set[WebSocket] = set()
        # Per-connection session subscription (None => no session bound yet).
        self._ws_sessions: dict[Any, str | None] = {}
        # Anti-CSWSH: browsers always send Origin; cross-origin handshakes are
        # rejected. Optional shared token via PYHARNESS_WEB_TOKEN.
        self.allowed_origins = {
            o.strip()
            for o in os.getenv("PYHARNESS_ALLOWED_ORIGINS", "").split(",")
            if o.strip()
        } or {
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
        }
        self.auth_token: str | None = os.getenv("PYHARNESS_WEB_TOKEN") or None
        self._session_tasks: dict[str, asyncio.Task] = {}
        self._static_dir = "static"

        self._setup_routes()

    def _setup_routes(self) -> None:
        """注册路由（避免在 __init__ 中使用 self 作为装饰器参数）。"""

        @self.app.post("/api/chat")
        async def chat(request: ChatRequest) -> dict[str, Any]:
            return await self._chat(request)

        @self.app.get("/api/sessions")
        async def list_sessions(namespace: str = "default", limit: int = 50, offset: int = 0) -> dict[str, Any]:
            return await self._list_sessions(namespace, limit, offset)

        @self.app.get("/api/sessions/{session_id}")
        async def get_session(session_id: str) -> dict[str, Any]:
            return await self._get_session(session_id)

        @self.app.delete("/api/sessions/{session_id}")
        async def delete_session(session_id: str) -> dict[str, Any]:
            return await self._delete_session(session_id)

        @self.app.patch("/api/sessions/{session_id}")
        async def rename_session(session_id: str, request: dict[str, Any]) -> dict[str, Any]:
            return await self._rename_session(session_id, request)

        @self.app.delete("/api/sessions")
        async def clear_sessions() -> dict[str, Any]:
            return await self._clear_sessions()

        @self.app.get("/api/plans/{plan_id}")
        async def get_plan(plan_id: str) -> dict[str, Any]:
            return await self._get_plan(plan_id)

        @self.app.post("/api/search")
        async def search(request: SearchRequest) -> dict[str, Any]:
            return await self._search(request)

        @self.app.websocket("/ws/events")
        async def websocket_events(ws: WebSocket) -> None:
            await self._websocket_events(ws)

        @self.app.get("/api/plugins")
        async def list_plugins() -> dict[str, Any]:
            return await self._list_plugins()

        @self.app.post("/api/plugins/load")
        async def load_plugin(request: PluginLoadRequest) -> dict[str, Any]:
            return await self._load_plugin(request)

        @self.app.post("/api/plugins/reload")
        async def reload_plugin(request: PluginReloadRequest) -> dict[str, Any]:
            return await self._reload_plugin(request)

        @self.app.delete("/api/plugins/{name}")
        async def unload_plugin(name: str) -> dict[str, Any]:
            return await self._unload_plugin(name)

        @self.app.get("/")
        async def index() -> FileResponse:
            return FileResponse(f"{self._static_dir}/index.html")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @hookimpl
    def harness_initialized(self, harness: Any) -> None:
        """Capture harness reference after entry-points are loaded."""
        self.harness = harness

    # ------------------------------------------------------------------
    # REST API handlers
    # ------------------------------------------------------------------
    async def _get_model(self, requested: str | None = None) -> str:
        """Resolve the best available model name."""
        providers = await _settle(
            self.harness.bus.pm.hook.get_llm_providers(context=None)
        )
        models: list[str] = []
        for value in providers:
            if value is not None:
                models.extend(value if isinstance(value, tuple) else (value,))
        if not models:
            return requested or "default"
        if requested and requested in models:
            return requested
        return models[0]

    async def _chat(self, request: ChatRequest) -> dict[str, Any]:
        """发送消息并启动 Agent Loop（非流式，返回最终结果）。"""
        if self.harness is None:
            return {"error": "Harness 未初始化"}

        model = await self._get_model(request.model)
        agent = AgentConfig(name="web-chat", model=model)
        try:
            ctx = await self.harness.run_session(
                agent=agent,
                initial_text=request.content,
                resume_session_id=request.session_id,
            )
            return {
                "session_id": str(ctx.session_id),
                "status": "ok",
                "message_count": len(ctx.messages),
            }
        except Exception as exc:
            logger.exception("chat failed")
            return {"error": str(exc), "status": "error"}

    # ------------------------------------------------------------------
    # Plugin lifecycle (hot-reload) handlers
    # ------------------------------------------------------------------
    async def _list_plugins(self) -> dict[str, Any]:
        if self.harness is None:
            return {"error": "Harness 未初始化"}
        try:
            plugins = await self.harness.list_plugins()
            return {"plugins": plugins}
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_plugins failed")
            return {"error": str(exc)}

    async def _load_plugin(self, request: PluginLoadRequest) -> dict[str, Any]:
        if self.harness is None:
            return {"error": "Harness 未初始化", "ok": False}
        try:
            result = await self.harness.load_plugin(request.path)
            await self.broadcast(
                "plugin_loaded",
                {"name": result["name"], "core": result.get("core", False)},
            )
            return result
        except PluginError as exc:
            logger.exception("load_plugin failed")
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("load_plugin failed")
            return {"ok": False, "error": str(exc)}

    async def _reload_plugin(self, request: PluginReloadRequest) -> dict[str, Any]:
        if self.harness is None:
            return {"error": "Harness 未初始化", "ok": False}
        try:
            result = await self.harness.reload_plugin(request.name)
            await self.broadcast(
                "plugin_loaded",
                {"name": result["name"], "core": result.get("core", False)},
            )
            return result
        except PluginError as exc:
            logger.exception("reload_plugin failed")
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("reload_plugin failed")
            return {"ok": False, "error": str(exc)}

    async def _unload_plugin(self, name: str) -> dict[str, Any]:
        if self.harness is None:
            return {"error": "Harness 未初始化", "ok": False}
        try:
            result = await self.harness.unload_plugin(name)
            await self.broadcast("plugin_unloaded", {"name": name})
            return result
        except PluginCoreProtectedError:
            return {"ok": False, "error": f"核心插件 '{name}' 不可卸载"}
        except PluginNotFoundError:
            return {"ok": False, "error": f"插件 '{name}' 不存在"}
        except PluginError as exc:
            logger.exception("unload_plugin failed")
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("unload_plugin failed")
            return {"ok": False, "error": str(exc)}

    async def _list_sessions(self, namespace: str, limit: int, offset: int) -> dict[str, Any]:
        """列出所有持久化的会话。"""
        sessions: list[SessionData] = []
        for value in await _settle(
            self.harness.bus.pm.hook.list_sessions(namespace=namespace, limit=limit, offset=offset)
        ):
            if value is not None:
                sessions.extend(value if isinstance(value, list) else [value])

        return {
            "sessions": [
                {
                    "session_id": str(s.session_id),
                    "namespace": s.namespace,
                    "created_at": s.created_at.isoformat(),
                    # Prefer the precomputed count; fall back for other stores.
                    "message_count": getattr(s, "message_count", 0) or len(s.messages),
                }
                for s in sessions
            ]
        }

    async def _get_session(self, session_id: str) -> dict[str, Any]:
        """获取会话详情（消息历史）。"""
        for value in await _settle(
            self.harness.bus.pm.hook.load_session(session_id=session_id)
        ):
            if value is not None:
                data = SessionData.model_validate(value)
                return {
                    "session_id": str(data.session_id),
                    "namespace": data.namespace,
                    "messages": [m.model_dump(mode="json") for m in data.messages],
                    "memory": data.memory,
                    "created_at": data.created_at.isoformat(),
                }
        return {"error": "Session not found"}

    async def _delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete a persisted session."""
        for value in await _settle(
            self.harness.bus.pm.hook.delete_session(session_id=session_id)
        ):
            if value is not None:
                return {"deleted": bool(value)}
        return {"deleted": False, "error": "Session not found"}

    async def _rename_session(self, session_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Rename a persisted session."""
        title = str(request.get("title", "")).strip()
        if not title:
            return {"error": "title is required"}
        for value in await _settle(
            self.harness.bus.pm.hook.rename_session(session_id=session_id, title=title)
        ):
            if value is not None:
                return {
                    "session_id": str(value.session_id),
                    "title": title,
                    "namespace": value.namespace,
                    "message_count": len(value.messages),
                }
        return {"error": "Session not found"}

    async def _clear_sessions(self) -> dict[str, Any]:
        """Delete all persisted sessions."""
        for value in await _settle(
            self.harness.bus.pm.hook.clear_sessions()
        ):
            if value is not None:
                return {"status": "ok", "cleared_count": int(value)}
        return {"status": "error", "cleared_count": 0}

    async def _get_plan(self, plan_id: str) -> dict[str, Any]:
        """获取计划状态。"""
        for value in await _settle(
            self.harness.bus.pm.hook.load_plan(plan_id=plan_id)
        ):
            if value is not None:
                plan = value
                return {
                    "plan_id": str(plan.plan_id),
                    "task": plan.task,
                    "status": plan.status,
                    "progress": plan.progress,
                    "steps": [s.model_dump(mode="json") for s in plan.steps],
                }
        return {"error": "Plan not found"}

    async def _search(self, request: SearchRequest) -> dict[str, Any]:
        """FTS5 全文检索历史会话。"""
        for value in await _settle(
            self.harness.bus.pm.hook.search_session(
                session_id=request.session_id,
                query=request.query,
                limit=request.limit,
            )
        ):
            if value is not None:
                results = value if isinstance(value, list) else [value]
                return {
                    "query": request.query,
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
        return {"query": request.query, "count": 0, "results": []}

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # WebSocket authorization (anti cross-site WebSocket hijacking)
    # ------------------------------------------------------------------
    def _ws_authorized(self, origin: str | None, token: str | None) -> bool:
        """Authorize an incoming WebSocket handshake.

        * Browsers always send ``Origin`` — when present it must be in the
          allowlist, otherwise the handshake is rejected (CSWSH defence).
        * When ``PYHARNESS_WEB_TOKEN`` is configured, a matching token
          (query param ``?token=`` or header ``X-PyHarness-Token``) is required.
        * Non-browser clients that omit ``Origin`` are allowed (local tooling),
          unless a token is configured — then the token is mandatory.
        """
        if self.auth_token is not None and token != self.auth_token:
            return False
        if origin and origin not in self.allowed_origins:
            return False
        return True

    async def _websocket_events(self, ws: WebSocket) -> None:
        """WebSocket 端点：实时推送 observe_event 事件。"""
        origin = ws.headers.get("origin")
        token = ws.query_params.get("token") or ws.headers.get("x-pyharness-token")
        if not self._ws_authorized(origin, token):
            logger.warning("WebSocket handshake rejected (origin=%r)", origin)
            await ws.accept()
            await ws.close(code=1008)  # policy violation
            return

        await ws.accept()
        self.ws_clients.add(ws)
        self._ws_sessions[ws] = None
        conn_id = str(uuid.uuid4())
        logger.info("WebSocket client connected: %s", conn_id)

        current_task: asyncio.Task | None = None

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "user_message":
                    if current_task is not None and not current_task.done():
                        current_task.cancel()
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass

                    content = msg.get("content", "")
                    session_id = msg.get("session_id")
                    model = msg.get("model")
                    self._ws_sessions[ws] = session_id
                    current_task = asyncio.create_task(
                        self._run_stream_session(session_id, content, model=model, ws=ws)
                    )

                elif msg_type == "cancel":
                    if current_task is not None and not current_task.done():
                        current_task.cancel()
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass
                        await self._send(ws, "session_end", {"status": "cancelled"})

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected: %s", conn_id)
        except Exception as exc:
            logger.error("WebSocket error: %s", exc)
        finally:
            self.ws_clients.discard(ws)
            self._ws_sessions.pop(ws, None)
            if current_task is not None and not current_task.done():
                current_task.cancel()

    async def _run_stream_session(
        self,
        session_id: str | None,
        content: str,
        model: str | None = None,
        ws: Any = None,
    ) -> None:
        """在后台运行流式会话，将 LLM 分片**单播**给发起请求的连接。

        事件只发给发起者（而不是广播给所有连接），避免多用户串流；事件负载
        携带 ``session_id``，流结束时回填真实会话 id 供前端续聊。
        """
        if self.harness is None:
            await self._send(ws, "error", {"message": "Harness 未初始化"})
            return

        resolved_model = await self._get_model(model)
        agent = AgentConfig(name="web-chat", model=resolved_model)
        try:
            async for chunk in self.harness.stream_session(
                agent=agent,
                initial_text=content,
                resume_session_id=session_id,
            ):
                await self._send(ws, "llm_stream_chunk", {
                    "session_id": session_id,
                    "content": chunk.delta,
                    "is_finished": False,
                    "tool_calls": [tc.model_dump(mode="json") for tc in chunk.tool_calls],
                })

            # 回填真实会话 id（新建会话时前端据此获得后续 resume 用的 id）。
            final_sid = session_id
            ctx = getattr(self.harness, "last_context", None)
            if ctx is not None:
                final_sid = str(ctx.session_id)
                if ws is not None and session_id is None:
                    self._ws_sessions[ws] = final_sid
            await self._send(ws, "llm_stream_chunk", {
                "session_id": final_sid,
                "content": "",
                "is_finished": True,
            })

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("stream_session failed")
            await self._send(ws, "error", {"message": str(exc)})

    # ------------------------------------------------------------------
    # Broadcast / unicast
    # ------------------------------------------------------------------
    def _serialize_event(self, event: str, data: dict[str, Any]) -> str | None:
        """Serialize one event payload; returns ``None`` on failure."""

        def _json_default(o: Any) -> Any:
            if hasattr(o, "model_dump"):
                return o.model_dump(mode="json")
            if hasattr(o, "to_text"):
                return o.to_text()
            if isinstance(o, uuid.UUID):
                return str(o)
            if hasattr(o, "value"):
                return o.value
            return str(o)

        def _serialize(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="json")
            if hasattr(obj, "to_text"):
                return obj.to_text()
            if hasattr(obj, "value"):
                return obj.value
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialize(v) for v in obj]
            if isinstance(obj, uuid.UUID):
                return str(obj)
            return obj

        try:
            payload = _serialize(data)
            return json.dumps({"type": event, "data": payload}, ensure_ascii=False, default=_json_default)
        except Exception as exc:
            logger.error("event serialization failed: %s", exc, exc_info=True)
            return None

    async def _send(self, ws: Any, event: str, data: dict[str, Any]) -> None:
        """Unicast one event to a single client (no-op when ``ws is None``).

        A failed send drops the client from the registry so dead sockets
        cannot accumulate or slow everyone else down.
        """
        if ws is None:
            return
        message = self._serialize_event(event, data)
        if message is None:
            return
        try:
            await ws.send_text(message)
        except Exception:
            self.ws_clients.discard(ws)
            self._ws_sessions.pop(ws, None)

    async def broadcast(self, event: str, data: dict[str, Any]) -> None:
        """向所有 WebSocket 客户端**并发**广播（互不阻塞；慢/断开的连接被移除）。"""
        if not self.ws_clients:
            return
        message = self._serialize_event(event, data)
        if message is None:
            return

        async def _push(ws: WebSocket) -> None:
            try:
                await ws.send_text(message)
            except Exception:
                self.ws_clients.discard(ws)
                self._ws_sessions.pop(ws, None)

        await asyncio.gather(*(_push(ws) for ws in list(self.ws_clients)))

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------
    def mount_static(self) -> None:
        """挂载静态文件目录。"""
        import os
        static_path = os.path.join(os.getcwd(), "static")
        if os.path.isdir(static_path):
            self.app.mount("/static", StaticFiles(directory=static_path), name="static")


# ---------------------------------------------------------------------------
# Serve helper
# ---------------------------------------------------------------------------
def serve(harness: Any, host: str = "127.0.0.1", port: int = 3080) -> None:
    """启动 Web UI 服务。"""
    import uvicorn
    from pyharness.plugins.tool_web import WebPlugin
    from pyharness.plugins.ui_websocket import WebSocketObserverPlugin
    from pyharness.plugins.tool_fs import FileSystemPlugin
    from pyharness.plugins.workflow import WorkflowPlugin
    from pyharness.plugins.tool_subagent import SubagentToolPlugin
    from pyharness.plugins.tool_python_exec import PythonExecPlugin
    from pyharness.plugins.session_store import SQLiteSessionStorePlugin

    harness.register_plugin(WebPlugin())
    harness.register_plugin(FileSystemPlugin())
    harness.register_plugin(WorkflowPlugin())
    harness.register_plugin(SubagentToolPlugin())
    harness.register_plugin(PythonExecPlugin())
    harness.register_plugin(SQLiteSessionStorePlugin())
    plugin = WebUIPlugin(host=host, port=port)
    harness.register_plugin(plugin)
    harness.register_plugin(WebSocketObserverPlugin(plugin))
    harness.initialize()
    plugin.mount_static()
    uvicorn.run(plugin.app, host=host, port=port)


# ---------------------------------------------------------------------------
# Tests helper
# ---------------------------------------------------------------------------
async def _settle(values: tuple[Any, ...]) -> list[Any]:
    """Await any coroutines returned by async hookimpls, pass sync values through."""
    import inspect
    if inspect.isawaitable(values):
        values = (values,)
    if not values:
        return []
    settled: list[Any] = []
    for value in values:
        try:
            settled.append(await value if inspect.isawaitable(value) else value)
        except Exception as exc:
            logger = logging.getLogger("pyharness.web_ui")
            logger.error("Hook implementation raised an exception: %s", exc, exc_info=True)
    return settled


__all__ = ["WebUIPlugin", "WebSocketObserverPlugin", "serve"]
