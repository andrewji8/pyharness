"""Engine core: the EventBus (pluggy) and the Harness orchestrator.

Thin-Core principle
-------------------
The engine owns exactly three responsibilities, nothing more:

1. **Event Bus** -- a pluggy ``PluginManager`` dispatches every hook call to the
   registered plugin set. ``@hookspec``/``@hookimpl`` are the mechanism; plugins
   are just Python objects registered into the manager (entry-points auto-load).
2. **Plugin Loading** -- optional discovery from package entry-points (group
   ``pyharness.plugins``) plus explicit ``register()``.
3. **Context Management** -- the session runner opens a task-scoped
   :class:`SessionContext` and advances it through immutable snapshots.

Everything domain-specific (LLM providers, tool executors, CLIs, Web UI) lives
in plugins and reaches the harness only through the hooks in :mod:`specs`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import sys
import time
import types
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from pluggy import PluginManager

from pyharness.context import SessionContext, _subagent_depth, current_context, open_session
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
    SessionData,
    SubagentResult,
    SubagentSpec,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
    ToolStreamEvent,
)
from pyharness.specs import AgentHooks


async def _settle(values: tuple[Any, ...]) -> list[Any]:
    """Await any coroutines returned by async hookimpls, pass sync values through.

    pluggy returns whichever objects its ``@hookimpl`` bodies return. For async
    impls that's a coroutine; for sync impls it's the plain value. This helper
    normalizes the mix into concrete results. ``None`` means no plugin
    implemented the hook (a firstresult hook with zero impls returns ``None``).
    A bare awaitable means a single firstresult impl returned a coroutine.

    Exceptions raised by individual hook implementations are logged and
    swallowed so that one misbehaving plugin cannot crash the agent loop.
    """
    if inspect.isawaitable(values):
        values = (values,)
    if not values:
        return []
    settled: list[Any] = []
    for value in values:
        try:
            settled.append(await value if inspect.isawaitable(value) else value)
        except Exception as exc:
            logger = logging.getLogger("pyharness.core")
            logger.error("Hook implementation raised an exception: %s", exc, exc_info=True)
    return settled


def _safe_hook(label: str, call: Callable[[], Any]) -> None:
    """Invoke a *synchronous* hook fan-out, isolating plugin failures.

    ``_settle`` already protects async hooks; a raising sync ``@hookimpl``
    would otherwise propagate straight through and kill the whole session.
    """
    try:
        call()
    except Exception:
        logger.error("sync hook %s raised; plugin failure isolated", label, exc_info=True)


def _distributed_enabled() -> bool:
    """Best-effort check for distributed tool execution.

    Delegates to :mod:`pyharness.distributed` so the Redis queue protocol and
    the environment switch live in one place.
    """
    from pyharness import distributed

    return distributed.distributed_exec_enabled()


def _tool_result_to_text(result: "ToolResult") -> str:
    """把 :class:`ToolResult` 规整为写入对话 transcript 的纯文本。

    ``ToolResult.output`` 是 ``dict[str, Any]``，而 ``Message.content`` 必须是
    字符串；不同工具把面向 LLM 的正文放在不同键上（``content``/``stdout``/
    ``text``/``echo``/``result``/``results``）。这里统一提取，丢弃易膨胀的
    ``code`` 字段，失败时直接返回 ``error`` 文案，避免把 Python dict repr
    直接塞进模型上下文。
    """
    if result.error:
        return result.error
    out = result.output
    if not out:
        return ""
    for key in ("content", "stdout", "text", "echo", "result"):
        value = out.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(v) for v in value)
    results = out.get("results")
    if isinstance(results, list):
        return "\n".join(str(v) for v in results)
    compact = {k: v for k, v in out.items() if k != "code"}
    if not compact:
        return ""
    try:
        return json.dumps(compact, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(compact)


def _tool_specs_to_openai_tools(specs: list[Any]) -> list[dict[str, Any]]:
    """Convert PyHarness ToolSpec list to OpenAI function-calling format."""
    tools: list[dict[str, Any]] = []
    for spec in specs:
        if not hasattr(spec, "name") or not hasattr(spec, "parameters"):
            continue
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in getattr(spec, "parameters", ()) or ():
            param_type = getattr(param, "type", "string") or "string"
            properties[param.name] = {
                "type": param_type,
                "description": getattr(param, "description", "") or "",
            }
            if getattr(param, "required", False):
                required.append(param.name)
        tool_def: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": getattr(spec, "description", "") or spec.name,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        tools.append(tool_def)
    return tools


# --------------------------------------------------------------------------- #
# Plugin lifecycle errors
# --------------------------------------------------------------------------- #
class PluginError(Exception):
    """Base error raised by plugin load/unload/reload operations."""


class PluginNotFoundError(PluginError):
    """Raised when a referenced plugin is not present in the registry."""


class PluginCoreProtectedError(PluginError):
    """Raised when an operation targets a protected core plugin."""


# Class names that are always treated as protected core plugins.
_CORE_PLUGIN_CLASS_NAMES = frozenset(
    {
        "SQLiteSessionStorePlugin",
        "HTTPProvider",
        "DummyProvider",
        "CliObserver",
    }
)


def _plugin_is_core(plugin: Any, name: str) -> bool:
    """Best-effort detection of whether a plugin is a protected core plugin.

    Core plugins are infrastructure the harness cannot run without (LLM
    providers, session store, the CLI observer). They must never be unloaded
    or reloaded at runtime.
    """
    if getattr(plugin, "CORE_PLUGIN", False):
        return True
    if isinstance(plugin, types.ModuleType):
        mod = plugin.__name__
        if mod.endswith("llm.entry") or "plugins.session_store" in mod:
            return True
    else:
        if type(plugin).__name__ in _CORE_PLUGIN_CLASS_NAMES:
            return True
    if name in _CORE_PLUGIN_CLASS_NAMES:
        return True
    return False


def _resolve_plugin_instance(module: types.ModuleType) -> Any:
    """Extract the plugin object from an imported module.

    Resolution order:

    1. ``module.PLUGIN`` -- a ready-made instance.
    2. ``module.create_plugin()`` -- a factory callable.
    3. The first class *defined in the module* that carries at least one
       ``@hookimpl`` method; it is instantiated.
    """
    plugin = getattr(module, "PLUGIN", None)
    if plugin is not None:
        return plugin

    factory = getattr(module, "create_plugin", None)
    if callable(factory):
        created = factory()
        if created is not None:
            return created

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if _class_has_hookimpl(obj):
            try:
                return obj()
            except Exception:
                continue
    return None


def _class_has_hookimpl(cls: type) -> bool:
    for attr in vars(cls).values():
        if callable(attr) and any(n.endswith("_hookimpl") for n in dir(attr)):
            return True
    return False


# ---------------------------------------------------------------------------
# Default system prompt (injected when agent.system_prompt is empty)
# ---------------------------------------------------------------------------
_DEFAULT_SYSTEM_PROMPT_TEMPLATE = """\
你是 PyHarness，一个具备联网能力的 AI Agent。你拥有以下工具：

{工具能力描述}

核心规则：
1. 当用户询问新闻、天气、股价、最新事件等任何实时信息时，你必须先调用联网工具获取数据，再基于工具返回的结果回答。
2. 当被问"你能联网吗"时，回答："可以，我能通过工具获取实时信息。"
3. 严禁直接回答"我无法联网"或"我没有联网能力"。
4. 优先使用工具获取准确信息，而不是依赖训练数据中的过期信息。
5. 当任务包含 2 个以上步骤时，必须先调用 workflow 工具创建执行计划，再按计划逐步执行。
6. 当子 Agent 未返回有效结果时，你必须自己调用工具真实执行，严禁凭记忆编造结果。"""


def _build_default_system_prompt(harness: "Harness", ctx: SessionContext, agent: AgentConfig) -> str:
    """Dynamically build a system prompt that lists available tools.

    Queries ``get_tool_specs`` to discover all registered tools, then formats
    them into the prompt so the LLM knows what it can call.
    """
    try:
        specs: dict[str, Any] = {}
        for plugin_specs in harness.bus.pm.hook.get_tool_specs(context=ctx):
            if plugin_specs is not None:
                specs.update({s.name: s for s in plugin_specs})

        if not specs:
            return "你是 PyHarness，一个 AI Agent。"

        tool_lines: list[str] = []
        for name, spec in specs.items():
            params = ", ".join(
                f"{p.name} ({p.type})" + (" *" if p.required else "")
                for p in getattr(spec, "parameters", ())
            )
            desc = getattr(spec, "description", "") or name
            tool_lines.append(f"- `{name}`: {desc} [参数: {params}]" if params else f"- `{name}`: {desc}")

        tool_text = "\n".join(tool_lines)
        return _DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(工具能力描述=tool_text)
    except Exception:
        return "你是 PyHarness，一个具备联网能力的 AI Agent。"


class EventBus:
    """Thin wrapper over a pluggy ``PluginManager`` (the event bus).

    ``hooks`` exposes the configured hook relay; ``register`` / ``on`` let both
    formal plugins and ephemeral listeners subscribe. ``emit`` / ``aemit``
    broadcast to ad-hoc listeners first, then fan out to every plugin through
    the ``observe`` hook.
    """

    def __init__(
        self,
        project_name: str = "pyharness",
        hook_spec: type = AgentHooks,
    ) -> None:
        self.project_name = project_name
        self.pm = PluginManager(project_name)
        self.pm.add_hookspecs(hook_spec)
        self._subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    # -- pluggy-facing API -------------------------------------------------- #
    @property
    def hooks(self) -> Any:
        """The configured hook relay (pass it to ``emit*`` or call directly)."""
        return self.pm.hook

    def register(self, plugin: Any, name: str | None = None) -> str | None:
        """Register a plugin object (its ``@hookimpl`` marks activate)."""
        return self.pm.register(plugin, name=name)

    def unregister(self, plugin: Any | None = None, name: str | None = None) -> Any | None:
        """Remove a previously registered plugin."""
        return self.pm.unregister(plugin=plugin, name=name)

    def load_entry_points(self, group: str = "pyharness.plugins") -> None:
        """Auto-load installed packages that expose ``group`` entry-points."""
        self.pm.load_setuptools_entrypoints(group)

    # -- ad-hoc (Cordis-style ``on``/``emit``) subscriptions ----------------- #
    def on(self, event_type: str, fn: Callable[..., Any]) -> Callable[[], None]:
        """Subscribe ``fn`` to ``event_type``. Returns an unsubscribe callable."""
        self._subscribers[event_type].append(fn)

        def unsubscribe() -> None:
            bucket = self._subscribers.get(event_type)
            if bucket and fn in bucket:
                bucket.remove(fn)

        return unsubscribe

    async def aemit(self, event_type: str, **payload: Any) -> list[Any]:
        """Broadcast asynchronously (awaiting any coroutine listener results)."""
        results: list[Any] = []
        for fn in list(self._subscribers.get(event_type, ())):
            value = fn(event_type=event_type, **payload)
            results.append(await value if inspect.isawaitable(value) else value)
        # Fan out to every plugin's observe hook as well (with a synthetic Event).
        ctx = payload.get("context")
        if ctx is not None:
            results.extend(
                await _settle(
                    self.pm.hook.observe(
                        context=ctx,
                        event=Event(type=event_type, session_id=ctx.session_id, payload={k: v for k, v in payload.items() if k != "context"}),
                    )
                )
            )
        return results


class Harness:
    """The thin-core orchestrator owning the event bus and session lifecycle."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.bus = event_bus or EventBus()
        self.config = config or HarnessConfig()
        # Terminal snapshot of the most recent stream_session() run.
        self.last_context: SessionContext | None = None
        # Live registry of every plugin known to the harness. Maps the
        # pluggy-assigned name to bookkeeping used by hot-reload.
        self._plugin_registry: dict[str, dict[str, Any]] = {}
        self._plugin_lock = asyncio.Lock()
        if self.config.auto_load_entry_points:
            self.bus.load_entry_points(self.config.plugin_group)
        self._record_pm_plugins()

    # -- lifecycle ---------------------------------------------------------- #
    def initialize(self) -> None:
        """Signal a newly-wired harness; best-effort sync fan-out to plugins."""
        _safe_hook("harness_initialized", lambda: self.bus.pm.hook.harness_initialized(harness=self))

    def shutdown(self) -> None:
        """Graceful teardown; best-effort sync fan-out to plugins."""
        _safe_hook("harness_shutdown", lambda: self.bus.pm.hook.harness_shutdown(harness=self))

    # -- plugin management -------------------------------------------------- #
    def register_plugin(self, plugin: Any, name: str | None = None, *, core: bool = False) -> str | None:
        """Register a plugin object with the harness.

        This is the preferred public API for adding plugins. It delegates to
        :meth:`EventBus.register`, records the plugin in the live registry, and
        returns the pluggy-assigned name (or ``None`` if registration failed).

        Args:
            plugin: The plugin module or instance to register.
            name: Optional explicit registry name. When omitted, a name is
                derived from the plugin (its class name, or ``module.__name__``).
            core: Force ``core=True``. Core plugins are protected from
                unload/reload at runtime.
        """
        assigned = self.bus.register(plugin, name=name)
        if assigned is not None:
            self._plugin_registry[assigned] = {
                "instance": plugin,
                "source_path": getattr(plugin, "__file__", None),
                "core": bool(core) or _plugin_is_core(plugin, assigned),
                "module_name": getattr(plugin, "__name__", None)
                if isinstance(plugin, types.ModuleType)
                else None,
            }
        return assigned

    def unregister_plugin(self, plugin: Any | None = None, name: str | None = None) -> Any | None:
        """Remove a previously registered plugin and drop it from the registry."""
        removed = self.bus.unregister(plugin=plugin, name=name)
        if name is not None and name in self._plugin_registry:
            self._plugin_registry.pop(name, None)
        return removed

    def _record_pm_plugins(self) -> None:
        """Mirror every pluggy-registered plugin into ``self._plugin_registry``.

        Idempotent: plugins already present (e.g. registered via
        :meth:`register_plugin`) keep their existing record.
        """
        for pname, plugin in self.bus.pm.list_name_plugin():
            if pname in self._plugin_registry:
                continue
            self._plugin_registry[pname] = {
                "instance": plugin,
                "source_path": getattr(plugin, "__file__", None),
                "core": _plugin_is_core(plugin, pname),
                "module_name": getattr(plugin, "__name__", None)
                if isinstance(plugin, types.ModuleType)
                else None,
            }

    async def list_plugins(self) -> list[dict[str, Any]]:
        """Return metadata for every registered plugin.

        Safe to call at any time; reads behind the plugin lock.
        """
        async with self._plugin_lock:
            return [
                {
                    "name": name,
                    "core": rec["core"],
                    "source_path": rec.get("source_path"),
                }
                for name, rec in self._plugin_registry.items()
            ]

    async def load_plugin(self, source: str) -> dict[str, Any]:
        """Load a plugin from a ``.py`` file at runtime.

        The module is imported via :mod:`importlib`, the plugin object is
        resolved (``PLUGIN`` / ``create_plugin()`` / first ``@hookimpl`` class),
        registered, and a ``plugin_loaded`` hook is broadcast. On any failure the
        partial import is rolled back (module dropped from ``sys.modules``,
        nothing registered) and a :class:`PluginError` is raised.

        Args:
            source: Path to the ``.py`` file exposing the plugin.

        Returns:
            ``{"ok": True, "name": <registry name>, "core": <bool>}``.

        Raises:
            PluginError: If the file is missing, cannot be imported, or contains
                no resolvable plugin object.
        """
        async with self._plugin_lock:
            return await self._do_load(source)

    async def unload_plugin(self, name: str) -> dict[str, Any]:
        """Unload a previously loaded plugin by registry name.

        Refuses to unload core plugins. Broadcasts ``plugin_unloaded`` before
        removing the plugin so it can release resources.

        Raises:
            PluginNotFoundError: If ``name`` is not registered.
            PluginCoreProtectedError: If ``name`` is a protected core plugin.
        """
        async with self._plugin_lock:
            return await self._do_unload(name)

    async def reload_plugin(self, name: str) -> dict[str, Any]:
        """Reload a file-backed plugin by registry name.

        Equivalent to ``unload`` + clearing the module cache + ``load``. Core
        plugins and plugins not backed by a file are refused.

        Returns:
            The result of the subsequent :meth:`load_plugin` call.

        Raises:
            PluginNotFoundError / PluginCoreProtectedError / PluginError.
        """
        async with self._plugin_lock:
            rec = self._plugin_registry.get(name)
            if rec is None:
                raise PluginNotFoundError(f"Plugin '{name}' not found")
            if rec["core"]:
                raise PluginCoreProtectedError(
                    f"Plugin '{name}' is a core plugin and cannot be reloaded"
                )
            source = rec.get("source_path")
            if not source:
                raise PluginError(
                    f"Plugin '{name}' was not loaded from a file; cannot reload"
                )
            await self._do_unload(name)
            module_name = rec.get("module_name")
            if module_name and module_name in sys.modules:
                sys.modules.pop(module_name, None)
            return await self._do_load(source)

    async def _do_load(self, source: str) -> dict[str, Any]:
        """Implementation of :meth:`load_plugin` (lock already held)."""
        path = Path(source)
        if not path.is_file():
            raise PluginError(f"Plugin file not found: {source}")

        module_name = f"pyharness_dynamic.{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise PluginError(f"Cannot import plugin module from {source}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - roll back and report
            sys.modules.pop(module_name, None)
            raise PluginError(f"Plugin import failed: {exc}") from exc

        instance = _resolve_plugin_instance(module)
        if instance is None:
            sys.modules.pop(module_name, None)
            raise PluginError(
                f"No plugin object in {source} "
                "(expected PLUGIN / create_plugin() / a class with @hookimpl)"
            )

        plugin_name = (
            getattr(module, "PLUGIN_NAME", None)
            or getattr(instance, "name", None)
            or path.stem
        )
        if plugin_name in self._plugin_registry:
            sys.modules.pop(module_name, None)
            raise PluginError(f"Plugin '{plugin_name}' is already loaded")

        try:
            self.bus.register(instance, name=plugin_name)
        except Exception as exc:  # noqa: BLE001 - roll back and report
            sys.modules.pop(module_name, None)
            raise PluginError(f"Plugin register failed: {exc}") from exc

        self._plugin_registry[plugin_name] = {
            "instance": instance,
            "source_path": str(path),
            "core": _plugin_is_core(instance, plugin_name),
            "module_name": module_name,
        }

        # Let the freshly loaded plugin capture the harness, and broadcast.
        try:
            _safe_hook(
                "harness_initialized",
                lambda: self.bus.pm.hook.harness_initialized(harness=self),
            )
        except Exception:  # noqa: BLE001
            logger.exception("plugin harness_initialized failed")
        try:
            await _settle(
                self.bus.pm.hook.plugin_loaded(harness=self, name=plugin_name)
            )
        except Exception:  # noqa: BLE001
            logger.exception("plugin_loaded hook failed")

        return {
            "ok": True,
            "name": plugin_name,
            "core": self._plugin_registry[plugin_name]["core"],
        }

    async def _do_unload(self, name: str) -> dict[str, Any]:
        """Implementation of :meth:`unload_plugin` (lock already held)."""
        rec = self._plugin_registry.get(name)
        if rec is None:
            raise PluginNotFoundError(f"Plugin '{name}' not found")
        if rec["core"]:
            raise PluginCoreProtectedError(
                f"Plugin '{name}' is a core plugin and cannot be unloaded"
            )

        try:
            await _settle(
                self.bus.pm.hook.plugin_unloaded(harness=self, name=name)
            )
        except Exception:  # noqa: BLE001
            logger.exception("plugin_unloaded hook failed")
        try:
            self.bus.unregister(name=name)
        except Exception:  # noqa: BLE001
            logger.exception("plugin unregister failed")

        self._plugin_registry.pop(name, None)
        return {"ok": True, "name": name}

    # -- session runner ----------------------------------------------------- #
    async def run_session(
        self,
        agent: AgentConfig,
        initial_text: str | None = None,
        *,
        initial_message: Message | None = None,
        continue_from: SessionContext | None = None,
        resume_session_id: str | None = None,
        namespace: str | None = None,
    ) -> SessionContext:
        """Open (or continue) an isolated context and drive the agent loop.

        Pass ``continue_from`` (a snapshot from a previous call) to keep a
        conversation on the same reversible lineage instead of starting fresh.
        Pass ``resume_session_id`` to attempt to restore a previously persisted
        session from the store plugins.

        Data flow (all reversibility-preserving snapshot transitions)::

            [optional] load_session(resume_session_id)
            open_session -> session_started
                         -> (optional) user message appended + observed
                         -> loop: agent_next
                                 |- execute_tool per tool_call (observed)
                                 |- assistant message appended
                                 |- [optional] save_session(after each turn)
                         -> check_shutdown / exit
                         -> session_finished
                         -> save_session(final)

        Returns the final :class:`SessionContext` snapshot (the terminal state).
        """
        resume_ctx = await self._resolve_resume_ctx(continue_from, resume_session_id)
        async with open_session(namespace=self.config.namespace, context=resume_ctx) as ctx:
            effective_agent, ctx = await self._begin_session(ctx, agent, initial_text, initial_message)

            out: list[SessionContext] = []
            async for _ in self._run_agent_loop(ctx, effective_agent, stream=False, out=out):
                pass
            ctx = out[0]

            _safe_hook("session_finished", lambda: self.bus.pm.hook.session_finished(context=ctx))
            self.last_context = ctx
            # Persist the final snapshot after the session ends.
            await self._save_session(ctx)
            return ctx

    async def _resolve_resume_ctx(
        self,
        continue_from: SessionContext | None,
        resume_session_id: str | None,
    ) -> SessionContext | None:
        """Restore a persisted session (by id) or adopt ``continue_from``."""
        resume_ctx = continue_from
        if resume_ctx is None and resume_session_id is not None:
            for value in await _settle(
                self.bus.pm.hook.load_session(session_id=resume_session_id)
            ):
                if value is not None:
                    data = SessionData.model_validate(value)
                    resume_ctx = SessionContext(
                        session_id=data.session_id,
                        branch_id=uuid.uuid4(),
                        parent_id=None,
                        namespace=data.namespace,
                        messages=data.messages,
                        memory=data.memory,
                        metadata={},
                        created_at=data.created_at,
                    )
                    break
        return resume_ctx

    async def _begin_session(
        self,
        ctx: SessionContext,
        agent: AgentConfig,
        initial_text: str | None,
        initial_message: Message | None = None,
    ) -> tuple[AgentConfig, SessionContext]:
        """Inject a default system prompt (if needed), fire ``session_started``
        and append the initial user message. Returns ``(effective_agent, ctx)``.
        """
        effective_agent = agent
        if not agent.system_prompt:
            default_prompt = _build_default_system_prompt(self, ctx, agent)
            if default_prompt:
                effective_agent = agent.model_copy(update={"system_prompt": default_prompt})

        _safe_hook("session_started", lambda: self.bus.pm.hook.session_started(context=ctx, agent=effective_agent))

        if initial_message is not None:
            ctx = ctx.append_message(initial_message)
            await self.bus.aemit(
                EventType.USER_MESSAGE.value,
                context=ctx,
                text=initial_message.content or "",
                parts=[p.model_dump(mode="json") for p in initial_message.parts],
            )
        elif initial_text is not None:
            ctx = ctx.append_message(Message(role=Role.USER, content=initial_text))
            await self.bus.aemit(
                EventType.USER_MESSAGE.value,
                context=ctx,
                text=initial_text,
            )
        return effective_agent, ctx

    async def _save_session(self, ctx: SessionContext) -> None:
        """Best-effort persistence: fan out to all ``save_session`` impls."""
        try:
            await _settle(self.bus.pm.hook.save_session(session=ctx))
        except Exception as exc:
            logging.getLogger("pyharness.core").error(
                "save_session failed: %s", exc, exc_info=True
            )

    async def stream_session(
        self,
        agent: AgentConfig,
        initial_text: str | None = None,
        *,
        initial_message: Message | None = None,
        continue_from: SessionContext | None = None,
        resume_session_id: str | None = None,
        namespace: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Same loop as :meth:`run_session` but streams assistant deltas.

        Pass ``continue_from`` (a snapshot) to continue an existing conversation
        on the same reversible lineage. Pass ``resume_session_id`` to attempt
        to restore a previously persisted session from the store plugins.

        Yields :class:`~pyharness.schema.LLMStreamChunk` as providers produce
        them. Prefers the ``llm_stream`` hook; falls back to ``llm_complete``
        (emitting the whole content as one chunk) when no provider streams.
        The terminal :class:`SessionContext` snapshot is stored at
        ``self.last_context`` after the loop ends.
        """
        resume_ctx = await self._resolve_resume_ctx(continue_from, resume_session_id)
        async with open_session(namespace=namespace or self.config.namespace, context=resume_ctx) as ctx:
            effective_agent, ctx = await self._begin_session(ctx, agent, initial_text, initial_message)

            out: list[SessionContext] = []
            async for chunk in self._run_agent_loop(ctx, effective_agent, stream=True, out=out):
                yield chunk
            ctx = out[0]

            _safe_hook("session_finished", lambda: self.bus.pm.hook.session_finished(context=ctx))
            self.last_context = ctx
            await self._save_session(ctx)

    async def build_request(
        self,
        ctx: SessionContext,
        agent: AgentConfig,
        tools_override: list[Any] | None = None,
    ) -> LLMRequest:
        """Assemble a schema-driven :class:`LLMRequest` from the transcript.

        Pre-pends the agent's system prompt (when set) and freezes the current
        messages into an immutable request handed to the provider layer. This is
        the single place the engine translates reversible session state into a
        provider-shaped request. Before returning, the message list is passed
        through the ``build_request`` plugin hook so plugins may intercept and
        transform it (e.g., context compaction).
        """
        system = (
            [Message(role=Role.SYSTEM, content=agent.system_prompt)]
            if agent.system_prompt
            else []
        )
        if tools_override is not None:
            tool_specs = tools_override
        else:
            tool_specs = [
                s
                for specs in self.bus.pm.hook.get_tool_specs(context=ctx)
                if specs
                for s in specs
            ]
        request = LLMRequest(
            model=agent.model,
            messages=tuple((*system, *ctx.messages)),
            temperature=agent.temperature,
            tools=_tool_specs_to_openai_tools(tool_specs),
        )
        logger.info("[subagent] %s llm_request tools=%s", agent.name, [t.get("function", {}).get("name") for t in request.tools])
        # Allow plugins to intercept and transform the message list.
        modified = await _settle(
            self.bus.pm.hook.build_request(messages=list(request.messages))
        )
        for value in modified:
            if value is not None:
                request = request.model_copy(update={"messages": tuple(value)})
                break
        return request

    # -- internals ---------------------------------------------------------- #
    async def _run_agent_loop(
        self,
        ctx: SessionContext,
        agent: AgentConfig,
        *,
        stream: bool = False,
        tools_override: list[Any] | None = None,
        allowed_tools: list[str] | None = None,
        out: list[SessionContext] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Single source of truth for the agent loop (DRY).

        Shared by :meth:`run_session`, :meth:`stream_session` and the subagent
        fallback. It builds the next :class:`LLMResponse`, appends the assistant
        message (carrying ``tool_calls``) *before* the tool results, executes the
        tools, and advances the state machine. ``stream=True`` additionally
        yields :class:`LLMStreamChunk` deltas; ``stream=False`` runs silently.

        The terminal :class:`SessionContext` is appended to ``out`` (a
        caller-supplied list) so both generators and coroutines can retrieve it.
        """
        depth = _subagent_depth.get(None) or 0
        steps = 0
        while steps < agent.max_steps:
            if stream:
                request = await self.build_request(ctx, agent, tools_override=tools_override)
                resp, chunks = await self._stream_turn(ctx, request)
                for chunk in chunks:
                    yield chunk
            else:
                resp = await self._agent_next(ctx, agent, tools_override=tools_override)

            if resp is None:
                break

            tool_specs = self._select_tool_specs(ctx, depth, tools_override, allowed_tools)

            # assistant(tool_calls) 必须在 tool 结果之前，否则 OpenAI 兼容接口
            # 会因 tool_call_id 无法关联而报错。
            if resp.tool_calls:
                ctx = ctx.append_message(
                    Message(role=Role.ASSISTANT, content=resp.content, tool_calls=resp.tool_calls)
                )
                ctx = await self._apply_tools(ctx, resp, tool_specs=tool_specs)
            elif resp.content:
                ctx = ctx.append_message(Message(role=Role.ASSISTANT, content=resp.content))

            if not resp.tool_calls:
                await self.bus.aemit(
                    EventType.ASSISTANT_FINISHED.value, context=ctx, text=resp.content
                )
                break
            if await self._should_stop(ctx, agent):
                break
            steps += 1

        if out is not None:
            out.append(ctx)

    def _select_tool_specs(
        self,
        ctx: SessionContext,
        depth: int,
        tools_override: list[Any] | None,
        allowed_tools: list[str] | None,
    ) -> dict[str, Any] | None:
        """Resolve the tool-spec map handed to ``_apply_tools``.

        Top-level sessions (no override, no whitelist) return ``None`` so
        ``_apply_tools`` fetches every registered spec. Subagent sessions pass a
        ``tools_override`` (inherited parent tools) and an optional
        ``allowed_tools`` whitelist; here we filter by whitelist and strip
        recursion-prone spawn tools at the current depth.
        """
        if tools_override is None and allowed_tools is None:
            return None
        if tools_override is not None:
            base = {t.name: t for t in tools_override}
        else:
            base = {}
            for plugin_specs in self.bus.pm.hook.get_tool_specs(context=ctx):
                base.update({s.name: s for s in plugin_specs})
        filtered = self._filter_tools(list(base.values()), allowed=allowed_tools, depth=depth)
        return {t.name: t for t in filtered}

    async def _agent_next(
        self,
        ctx: SessionContext,
        agent: AgentConfig,
        tools_override: list[Any] | None = None,
    ) -> LLMResponse | None:
        """Decide the next assistant response.

        Tries the ``agent_next`` plugin hooks first (an optional policy
        override); when none responds, builds an :class:`LLMRequest` from the
        transcript and calls the provider layer via ``llm_complete``.
        """
        for value in await _settle(
            self.bus.pm.hook.agent_next(context=ctx, agent=agent)
        ):
            if value is not None:
                return LLMResponse.model_validate(value)
        request = await self.build_request(ctx, agent, tools_override=tools_override)
        return await self._complete(ctx, request)

    async def _complete(
        self, ctx: SessionContext, request: LLMRequest
    ) -> LLMResponse | None:
        """Non-streaming provider call: first non-``None`` response wins."""
        for value in await _settle(
            self.bus.pm.hook.llm_complete(context=ctx, request=request)
        ):
            if value is not None:
                return LLMResponse.model_validate(value)
        return None

    async def _stream_turn(
        self, ctx: SessionContext, request: LLMRequest
    ) -> tuple[LLMResponse | None, list[LLMStreamChunk]]:
        """Produce the next response, preferring a streaming provider.

        Returns ``(response, chunks)``. ``chunks`` are the raw deltas for UIs;
        ``response`` is the accumulated, tool-call-aware result the loop acts
        on. A stream that yields nothing (empty text, no tool calls) maps to
        ``(None, [...])`` so the loop can terminate cleanly.
        """
        carriers = [
            g for g in self.bus.pm.hook.llm_stream(context=ctx, request=request)
        ]
        if carriers:
            content: list[str] = []
            calls: list[Any] = []
            chunks: list[LLMStreamChunk] = []
            async for chunk in carriers[0]:
                content.append(chunk.delta)
                calls.extend(chunk.tool_calls)
                chunks.append(chunk)
            text = "".join(content)
            if not text and not calls:
                return None, chunks
            return (
                LLMResponse(model=request.model, content=text, tool_calls=tuple(calls)),
                chunks,
            )

        resp = await self._complete(ctx, request)
        if resp is None:
            return None, []
        return resp, ([LLMStreamChunk(delta=resp.content)] if resp.content else [])

    async def _apply_tools(
        self,
        ctx: SessionContext,
        resp: LLMResponse,
        tool_specs: dict[str, Any] | None = None,
    ) -> SessionContext:
        """Run every tool call in ``resp``, appending tool messages as snapshots."""
        if not resp.tool_calls:
            return ctx
        # get_tool_specs is a fan-in hook: each plugin contributes a tuple, so
        # flatten across all impls into a name->spec map.
        specs: dict[str, Any] = tool_specs if tool_specs is not None else {}
        if not specs:
            for plugin_specs in self.bus.pm.hook.get_tool_specs(context=ctx):
                specs.update({s.name: s for s in plugin_specs})
        for call in resp.tool_calls:
            spec = specs.get(call.tool_name)
            if spec is None:
                error_result = ToolResult(
                    tool_name=call.tool_name or "unknown",
                    status=ToolResultStatus.ERROR,
                    error=f"工具 '{call.tool_name}' 未注册或不存在。",
                )
                ctx = ctx.append_message(
                    Message(role=Role.TOOL, name=call.tool_name or "unknown", content=error_result.error, tool_call_id=call.id)
                )
                await self.bus.aemit(
                    EventType.TOOL_RESULT.value,
                    context=ctx,
                    tool=call.tool_name or "unknown",
                    result=error_result,
                )
                continue
            # 1. Give plugins a chance to intercept before actual execution.
            pre = await self._pre_tool_execute(ctx, call)
            if pre is not None:
                result = pre
            else:
                await self.bus.aemit(
                    EventType.TOOL_CALLED.value,
                    context=ctx,
                    tool=call.tool_name,
                    arguments=call.arguments,
                )
                result = await self._exec_tool(ctx, spec, call.arguments)
                logger.info("[subagent] tool_call=%s tool_result_status=%s", call.tool_name, result.status)
                await self.bus.aemit(
                    EventType.TOOL_RESULT.value,
                    context=ctx,
                    tool=call.tool_name,
                    result=result,
                )
            ctx = ctx.append_message(
                Message(role=Role.TOOL, name=call.tool_name, content=_tool_result_to_text(result), tool_call_id=call.id)
            )
        return ctx

    async def _pre_tool_execute(
        self, ctx: SessionContext, call: ToolCall
    ) -> ToolResult | None:
        """Call the ``pre_tool_execution`` hook; return the first non-None
        :class:`ToolResult` from any plugin, or ``None`` if no plugin intercepts."""
        for value in await _settle(
            self.bus.pm.hook.pre_tool_execution(tool_call=call)
        ):
            if value is not None:
                return ToolResult.model_validate(value)
        return None

    async def _exec_tool(
        self, ctx: SessionContext, spec: Any, arguments: dict[str, Any]
    ) -> ToolResult:
        """Invoke the tool executor hook, tolerating a ``None`` (no provider).

        A per-call ``tool_emitter`` closure is injected into the context so any
        long-running tool can stream intermediate output (stdout/stderr/...) to
        observers via the ``EventType.TOOL_STREAM`` event.

        When distributed execution is enabled (:func:`distributed_exec_enabled`),
        the tool is offloaded to a remote worker via a Redis queue instead of
        running in-process, and the result is awaited (with timeout).
        """
        if _distributed_enabled():
            return await self._exec_tool_distributed(ctx, spec, arguments)

        async def emitter(event: "ToolStreamEvent") -> None:
            await self.bus.aemit(
                EventType.TOOL_STREAM.value, context=ctx, event=event
            )

        ctx_with_emitter = ctx.model_copy(update={"tool_emitter": emitter})
        results = await _settle(
            self.bus.pm.hook.execute_tool(
                context=ctx_with_emitter, tool=spec, arguments=arguments
            )
        )
        for value in results:
            if value is not None:
                return ToolResult.model_validate(value)
        return ToolResult(tool_name=spec.name, status=ToolResultStatus.ERROR, error="no executor plugin")

    async def _exec_tool_distributed(
        self, ctx: SessionContext, spec: Any, arguments: dict[str, Any]
    ) -> ToolResult:
        """Enqueue the tool to a Redis task queue and await the worker result.

        The local emitter is still wired so that, while the worker runs, its
        ``tool.stream`` events stream back through the batch-2 Redis event bus
        and are re-injected here for the local observers — keeping the front-end
        console streaming without any new plumbing.
        """
        from pyharness import distributed

        async def emitter(event: "ToolStreamEvent") -> None:
            await self.bus.aemit(EventType.TOOL_STREAM.value, context=ctx, event=event)

        task = distributed.ToolTask(
            session_id=ctx.session_id,
            tool=ToolSpec.model_validate(spec) if not hasattr(spec, "model_dump") else spec,
            arguments=dict(arguments),
            timeout=getattr(spec, "timeout_seconds", 30.0),
            source_instance_id=os.getenv("PYHARNESS_INSTANCE_ID", ""),
        )

        import redis.asyncio as redis

        client = redis.from_url(os.getenv("REDIS_URL", ""), encoding="utf-8", decode_responses=True)
        try:
            await distributed.enqueue_tool_task(client, task)
            return await distributed.await_tool_result(client, task.task_id, timeout=task.timeout)
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    async def _should_stop(self, ctx: SessionContext, agent: AgentConfig) -> bool:
        """Ask plugins whether the loop should stop now (default: keep going)."""
        results = await _settle(
            self.bus.pm.hook.check_shutdown(context=ctx, agent=agent)
        )
        return any(bool(v) for v in results)

    # -- subagent orchestration ----------------------------------------------- #
    async def spawn_subagent(
        self,
        spec: SubagentSpec,
        parent_tools: list[ToolSpec] | None = None,
        parent_config: AgentConfig | None = None,
    ) -> SubagentResult:
        """Spawn a worker subagent to execute ``spec.task`` independently.

        This is the preferred entry point for orchestrator plugins. The engine
        enforces depth limits and timeouts, then delegates execution to the
        ``spawn_subagent`` plugin hook. If no plugin answers, the engine
        falls back to running a fresh :meth:`run_session` internally.

        Parameters
        ----------
        spec:
            The subagent specification (task, model, timeout, etc.).
        parent_tools:
            Tool specs inherited from the parent agent. The subagent can use
            this to know which tools are available. ``None`` means inherit all.
        parent_config:
            The parent agent's configuration, used for model negotiation and
            other shared settings. ``None`` means use defaults.
        """
        current_depth = _subagent_depth.get(None) or 0

        # Safety: enforce nesting depth limit.
        if current_depth >= 3:
            return SubagentResult(
                spec=spec,
                status="error",
                error=f"Subagent depth limit exceeded (max 3, current {current_depth}).",
                session_id=str(parent_config.session_id) if parent_config else "",
            )

        tools = parent_tools or []
        config = parent_config or AgentConfig(name="parent", model="default")

        # Give plugins first refusal.
        for value in await _settle(
            self.bus.pm.hook.spawn_subagent(spec=spec, parent_tools=tools, parent_config=config)
        ):
            if value is not None:
                result = SubagentResult.model_validate(value)
                await _settle(self.bus.pm.hook.subagent_finished(result=result))
                return result

        # Fallback: run a fresh session internally.
        return await self._run_subagent(spec, tools, config, current_depth)

    async def _run_subagent(
        self,
        spec: SubagentSpec,
        parent_tools: list[ToolSpec],
        parent_config: AgentConfig,
        depth: int,
    ) -> SubagentResult:
        """Internal fallback: run a sub-session with timeout and depth tracking."""
        start = time.time()
        child_session_id = uuid.uuid4()

        model = spec.model if spec.model != "default" else parent_config.model
        system_prompt = spec.system_prompt or (
            "你是一个子 Agent，正在执行上级指派的具体任务。"
            "你必须调用 python_exec 工具真实运行代码并基于输出回答，"
            "严禁心算或只写代码不运行。"
            "如需联网搜索，调用 web_search；如需读取文件，调用 fs_read。"
        )
        child_agent = AgentConfig(
            name=spec.name,
            model=model,
            system_prompt=system_prompt,
            max_steps=spec.max_turns,
        )

        token = _subagent_depth.set(depth + 1)
        try:
            parent = current_context()
            ns = self.config.namespace
            if parent is not None and parent.namespace.startswith("workflow-step:"):
                ns = parent.namespace
            async with open_session(namespace=ns) as ctx:
                await _settle(
                    self.bus.aemit(
                        "subagent_start",
                        context=ctx,
                        name=spec.name,
                        task=spec.task,
                    )
                )
                if spec.task:
                    ctx = ctx.append_message(Message(role=Role.USER, content=spec.task))
                logger.info("[subagent] %s entering loop allowed=%s parent_tools=%s", spec.name, spec.allowed_tools, [t.name for t in parent_tools])
                final = await asyncio.wait_for(
                    self._execute_agent_loop(
                        ctx,
                        child_agent,
                        allowed_tools=spec.allowed_tools,
                        parent_tools=parent_tools,
                    ),
                    timeout=spec.timeout,
                )
                duration = time.time() - start
                last = final.last_message.content if final.last_message else None
                result = SubagentResult(
                    spec=spec,
                    status="ok",
                    output=last,
                    duration_seconds=duration,
                    session_id=str(child_session_id),
                )
                await _settle(self.bus.pm.hook.subagent_finished(result=result))
                await _settle(
                    self.bus.aemit(
                        "subagent_complete",
                        context=ctx,
                        name=spec.name,
                        status="ok",
                        summary=last or "",
                    )
                )
                return result
        except TimeoutError:
            duration = time.time() - start
            result = SubagentResult(
                spec=spec,
                status="timeout",
                error=f"Subagent timed out after {spec.timeout}s",
                duration_seconds=duration,
                session_id=str(child_session_id),
            )
            await _settle(self.bus.pm.hook.subagent_finished(result=result))
            await _settle(
                self.bus.aemit(
                    "subagent_complete",
                    context=ctx,
                    name=spec.name,
                    status="timeout",
                    summary=result.error or "",
                )
            )
            return result
        except Exception as exc:
            duration = time.time() - start
            result = SubagentResult(
                spec=spec,
                status="error",
                error=str(exc),
                duration_seconds=duration,
                session_id=str(child_session_id),
            )
            await _settle(self.bus.pm.hook.subagent_finished(result=result))
            await _settle(
                self.bus.aemit(
                    "subagent_complete",
                    context=ctx,
                    name=spec.name,
                    status="error",
                    summary=result.error or "",
                )
            )
            return result
        finally:
            _subagent_depth.reset(token)

    @staticmethod
    def _filter_tools(
        tools: list[ToolSpec],
        allowed: list[str] | None,
        depth: int = 0,
        max_depth: int = 3,
    ) -> list[ToolSpec]:
        """Filter ``tools`` by an optional whitelist and enforce recursion guard.

        If ``allowed`` is ``None`` or empty, all tools are kept. When ``depth`` reaches
        ``max_depth``, subagent-spawning tools are removed to prevent runaway
        nesting.
        """
        if allowed:
            allowed_set = set(allowed)
            filtered = [t for t in tools if t.name in allowed_set]
        else:
            filtered = list(tools)

        if depth >= max_depth:
            filtered = [
                t for t in filtered
                if t.name not in ("spawn_subagent", "spawn_subagents")
            ]
        return filtered

    async def _execute_agent_loop(
        self,
        ctx: SessionContext,
        agent: AgentConfig,
        allowed_tools: list[str] | None = None,
        parent_tools: list[Any] | None = None,
    ) -> SessionContext:
        """Subagent loop: thin wrapper over :meth:`_run_agent_loop` (non-stream).

        Preserves the subagent-specific tool filtering (whitelist + recursion
        guard via ``_filter_tools``) and returns the terminal ``SessionContext``
        so :meth:`_run_subagent` can read the final message.
        """
        logger.info("[subagent] %s tools=%s", agent.name, [t.name for t in (parent_tools or [])])
        out: list[SessionContext] = []
        async for _ in self._run_agent_loop(
            ctx,
            agent,
            stream=False,
            tools_override=parent_tools,
            allowed_tools=allowed_tools,
            out=out,
        ):
            pass
        return out[0]
