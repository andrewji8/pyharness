"""Engine core: the EventBus (pluggy) and the Harness orchestrator.

Thin-Core principle
-------------------
The engine owns exactly three responsibilities, nothing more:

1. **Event Bus** — a pluggy ``PluginManager`` dispatches every hook call to the
   registered plugin set. ``@hookspec``/``@hookimpl`` are the mechanism; plugins
   are just Python objects registered into the manager (entry-points auto-load).
2. **Plugin Loading** — optional discovery from package entry-points (group
   ``pyharness.plugins``) plus explicit ``register()``.
3. **Context Management** — the session runner opens a task-scoped
   :class:`SessionContext` and advances it through immutable snapshots.

Everything domain-specific (LLM providers, tool executors, CLIs, Web UI) lives
in plugins and reaches the harness only through the hooks in :mod:`specs`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from pluggy import PluginManager

from pyharness.context import SessionContext, _subagent_depth, open_session
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
)
from pyharness.specs import AgentHooks

if TYPE_CHECKING:
    from pluggy._hooks import _HookRelay  # only for type hints


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
        if self.config.auto_load_entry_points:
            self.bus.load_entry_points(self.config.plugin_group)

    # -- lifecycle ---------------------------------------------------------- #
    def initialize(self) -> None:
        """Signal a newly-wired harness; best-effort sync fan-out to plugins."""
        self.bus.pm.hook.harness_initialized(harness=self)

    def shutdown(self) -> None:
        """Graceful teardown; best-effort sync fan-out to plugins."""
        self.bus.pm.hook.harness_shutdown(harness=self)

    # -- plugin management -------------------------------------------------- #
    def register_plugin(self, plugin: Any, name: str | None = None) -> str | None:
        """Register a plugin object with the harness.

        This is the preferred public API for adding plugins. It delegates to
        :meth:`EventBus.register` and returns the pluggy-assigned name, or
        ``None`` if registration failed.
        """
        return self.bus.register(plugin, name=name)

    def unregister_plugin(self, plugin: Any | None = None, name: str | None = None) -> Any | None:
        """Remove a previously registered plugin."""
        return self.bus.unregister(plugin=plugin, name=name)

    # -- session runner ----------------------------------------------------- #
    async def run_session(
        self,
        agent: AgentConfig,
        initial_text: str | None = None,
        *,
        continue_from: SessionContext | None = None,
        resume_session_id: str | None = None,
    ) -> SessionContext:
        """Open (or continue) an isolated context and drive the agent loop.

        Pass ``continue_from`` (a snapshot from a previous call) to keep a
        conversation on the same reversible lineage instead of starting fresh.
        Pass ``resume_session_id`` to attempt to restore a previously persisted
        session from the store plugins.

        Data flow (all reversibility-preserving snapshot transitions)::

            [optional] load_session(resume_session_id)
            open_session → session_started
                         → (optional) user message appended + observed
                         → loop: agent_next
                                 └─ execute_tool per tool_call (observed)
                                 └─ assistant message appended
                                 └─ [optional] save_session(after each turn)
                         → check_shutdown / exit
                         → session_finished
                         → save_session(final)

        Returns the final :class:`SessionContext` snapshot (the terminal state).
        """
        # 1. Try to resume from persistent store before opening a new session.
        resume_ctx: SessionContext | None = continue_from
        if resume_ctx is None and resume_session_id is not None:
            for value in await _settle(
                self.bus.pm.hook.load_session(session_id=resume_session_id)
            ):
                if value is not None:
                    data = SessionData.model_validate(value)
                    resume_ctx = SessionContext(
                        session_id=data.session_id,
                        branch_id=__import__("uuid").uuid4(),
                        parent_id=None,
                        namespace=data.namespace,
                        messages=data.messages,
                        memory=data.memory,
                        metadata={},
                        created_at=data.created_at,
                    )
                    break

        async with open_session(namespace=self.config.namespace, context=resume_ctx) as ctx:
            self.bus.pm.hook.session_started(context=ctx, agent=agent)

            if initial_text is not None:
                ctx = ctx.append_message(Message(role=Role.USER, content=initial_text))
                await self.bus.aemit(
                    EventType.USER_MESSAGE.value,
                    context=ctx,
                    text=initial_text,
                )

            steps = 0
            while steps < agent.max_steps:
                resp = await self._agent_next(ctx, agent)
                if resp is None:
                    break

                ctx = await self._apply_tools(ctx, resp)
                if resp.content:
                    ctx = ctx.append_message(
                        Message(role=Role.ASSISTANT, content=resp.content)
                    )

                if not resp.tool_calls:
                    await self.bus.aemit(
                        EventType.ASSISTANT_FINISHED.value,
                        context=ctx,
                        text=resp.content,
                    )
                    break

                if await self._should_stop(ctx, agent):
                    break
                steps += 1

            self.bus.pm.hook.session_finished(context=ctx)
            self.last_context = ctx

            # 2. Persist the final snapshot after the session ends.
            await self._save_session(ctx)

            return ctx

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
        continue_from: SessionContext | None = None,
        resume_session_id: str | None = None,
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
        resume_ctx: SessionContext | None = continue_from
        if resume_ctx is None and resume_session_id is not None:
            for value in await _settle(
                self.bus.pm.hook.load_session(session_id=resume_session_id)
            ):
                if value is not None:
                    data = SessionData.model_validate(value)
                    resume_ctx = SessionContext(
                        session_id=data.session_id,
                        branch_id=__import__("uuid").uuid4(),
                        parent_id=None,
                        namespace=data.namespace,
                        messages=data.messages,
                        memory=data.memory,
                        metadata={},
                        created_at=data.created_at,
                    )
                    break

        async with open_session(namespace=self.config.namespace, context=resume_ctx) as ctx:
            self.bus.pm.hook.session_started(context=ctx, agent=agent)

            if initial_text is not None:
                ctx = ctx.append_message(Message(role=Role.USER, content=initial_text))
                await self.bus.aemit(
                    EventType.USER_MESSAGE.value, context=ctx, text=initial_text
                )

            steps = 0
            while steps < agent.max_steps:
                request = await self.build_request(ctx, agent)
                resp, chunks = await self._stream_turn(ctx, request)
                for chunk in chunks:
                    yield chunk

                if resp is None:
                    break
                ctx = await self._apply_tools(ctx, resp)
                if resp.content:
                    ctx = ctx.append_message(
                        Message(role=Role.ASSISTANT, content=resp.content)
                    )
                if not resp.content and not resp.tool_calls:
                    break
                if not resp.tool_calls:
                    await self.bus.aemit(
                        EventType.ASSISTANT_FINISHED.value, context=ctx, text=resp.content
                    )
                    break
                if await self._should_stop(ctx, agent):
                    break
                steps += 1

            self.bus.pm.hook.session_finished(context=ctx)
            self.last_context = ctx
            await self._save_session(ctx)

    async def build_request(self, ctx: SessionContext, agent: AgentConfig) -> LLMRequest:
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
        request = LLMRequest(
            model=agent.model,
            messages=tuple((*system, *ctx.messages)),
            temperature=agent.temperature,
        )
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
    async def _agent_next(self, ctx: SessionContext, agent: AgentConfig) -> LLMResponse | None:
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
        request = await self.build_request(ctx, agent)
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
        # flatten across all impls into a name→spec map.
        specs: dict[str, Any] = tool_specs if tool_specs is not None else {}
        if not specs:
            for plugin_specs in self.bus.pm.hook.get_tool_specs(context=ctx):
                specs.update({s.name: s for s in plugin_specs})
        for call in resp.tool_calls:
            spec = specs.get(call.tool_name)
            if spec is None:
                ctx = ctx.append_message(
                    Message(role=Role.TOOL, name=call.tool_name, content="<unknown tool>")
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
                await self.bus.aemit(
                    EventType.TOOL_RESULT.value,
                    context=ctx,
                    tool=call.tool_name,
                    result=result,
                )
            ctx = ctx.append_message(
                Message(role=Role.TOOL, name=call.tool_name, content=str(result.output))
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
        """Invoke the tool executor hook, tolerating a ``None`` (no provider)."""
        results = await _settle(
            self.bus.pm.hook.execute_tool(context=ctx, tool=spec, arguments=arguments)
        )
        for value in results:
            if value is not None:
                return ToolResult.model_validate(value)
        return ToolResult(tool_name=spec.name, status=ToolResultStatus.ERROR, error="no executor plugin")

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
        import uuid

        start = __import__("time").time()
        child_session_id = uuid.uuid4()

        child_agent = AgentConfig(
            name=spec.name,
            model=spec.model,
            system_prompt=spec.system_prompt or "",
            max_steps=spec.max_turns,
        )

        token = _subagent_depth.set(depth + 1)
        try:
            async with open_session(namespace=self.config.namespace) as ctx:
                if spec.task:
                    ctx = ctx.append_message(Message(role=Role.USER, content=spec.task))
                final = await asyncio.wait_for(
                    self._execute_agent_loop(ctx, child_agent, allowed_tools=spec.allowed_tools),
                    timeout=spec.timeout,
                )
                duration = __import__("time").time() - start
                last = final.last_message.content if final.last_message else None
                result = SubagentResult(
                    spec=spec,
                    status="ok",
                    output=last,
                    duration_seconds=duration,
                    session_id=str(child_session_id),
                )
                await _settle(self.bus.pm.hook.subagent_finished(result=result))
                return result
        except TimeoutError:
            duration = __import__("time").time() - start
            result = SubagentResult(
                spec=spec,
                status="timeout",
                error=f"Subagent timed out after {spec.timeout}s",
                duration_seconds=duration,
                session_id=str(child_session_id),
            )
            await _settle(self.bus.pm.hook.subagent_finished(result=result))
            return result
        except Exception as exc:
            duration = __import__("time").time() - start
            result = SubagentResult(
                spec=spec,
                status="error",
                error=str(exc),
                duration_seconds=duration,
                session_id=str(child_session_id),
            )
            await _settle(self.bus.pm.hook.subagent_finished(result=result))
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

        If ``allowed`` is ``None``, all tools are kept. When ``depth`` reaches
        ``max_depth``, subagent-spawning tools are removed to prevent runaway
        nesting.
        """
        if allowed is not None:
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
    ) -> SessionContext:
        """Execute a stripped-down agent loop for subagents.

        This reuses the core loop logic but is simplified for the subagent
        context (no resume, no save hooks).

        **保留 vs 省略的行为：**
        - ✅ Guard 审批：通过 ``_apply_tools`` 中的 ``_pre_tool_execute`` 保留，
          子 Agent 执行高危工具同样需要人工确认。
        - ✅ 事件观测：通过 ``_agent_next`` / ``_apply_tools`` 中的
          ``observe_event`` 保留，UI 可展示子 Agent 进度。
        - ❌ Context Compaction：子 Agent 的 ``max_turns`` 通常很小（默认 5），
          不太可能触发 Token 爆炸，故省略 ``build_request`` 钩子调用。
        - ❌ spawn_subagents 工具：由 ``depth`` 控制递归，不在子 Agent 中
          注册并行派生工具，防止无限嵌套。
        """
        depth = _subagent_depth.get(None) or 0
        depth = _subagent_depth.get(None) or 0
        steps = 0
        while steps < agent.max_steps:
            resp = await self._agent_next(ctx, agent)
            if resp is None:
                break

            # 递归防护：根据当前深度过滤可用工具
            all_tools: dict[str, Any] = {}
            for plugin_specs in self.bus.pm.hook.get_tool_specs(context=ctx):
                all_tools.update({s.name: s for s in plugin_specs})
            filtered_tools = self._filter_tools(
                list(all_tools.values()),
                allowed=allowed_tools,
                depth=depth,
            )
            tool_map = {t.name: t for t in filtered_tools}

            ctx = await self._apply_tools(ctx, resp, tool_specs=tool_map)
            if resp.content:
                ctx = ctx.append_message(
                    Message(role=Role.ASSISTANT, content=resp.content)
                )
            if not resp.tool_calls:
                break
            if await self._should_stop(ctx, agent):
                break
            steps += 1
        return ctx