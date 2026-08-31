"""Workflow / Plan task orchestration plugin.

Provides the ``workflow_execute`` tool, allowing the LLM to generate a
structured execution plan (To-Do List) for a complex task and then execute it
step by step with dependency tracking and retry support.

Design
------
* The plugin captures the harness reference via ``harness_initialized`` so it
  can delegate to ``Harness.run_session`` for each step.
    * Planning uses the LLM directly via ``stream_session`` with a
      structured prompt that asks for JSON output.
* Execution respects step dependencies (topological order) and supports
  retries for failed steps.
* Each step runs in its own isolated session context via ``run_session``,
  keeping the workflow state cleanly separated from the parent conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from pluggy import HookimplMarker

from pyharness.context import SessionContext, current_context, set_current, reset_current
from pyharness.core import _settle
from pyharness.schema import (
    AgentConfig,
    LLMRequest,
    LLMResponse,
    Message,
    PlanAction,
    Role,
    StepStatus,
    SubagentResult,
    SubagentSpec,
    ToolArg,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
    UpdatePlanInput,
    WorkflowPlan,
    WorkflowStep,
    _utcnow,
)

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")


def _utc_now() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class WorkflowPlugin:
    """Tool provider for structured task planning and execution."""

    def __init__(self) -> None:
        self.harness: Any = None
        self._plans: dict[str, WorkflowPlan] = {}

    @hookimpl
    def harness_initialized(self, harness: Any) -> None:
        """Capture the harness reference after entry-points are loaded."""
        self.harness = harness

    @hookimpl
    def session_started(self, context: SessionContext, agent: AgentConfig | None = None) -> None:
        """Restore workflow plans from session memory on session start."""
        plans_memory = context.memory.get("workflow_plans", {})
        for plan_id, plan_data in plans_memory.items():
            try:
                plan = WorkflowPlan.model_validate(plan_data)
                self._plans[plan_id] = plan
            except Exception:
                pass

    async def _persist_plan(self, plan: WorkflowPlan) -> None:
        """Persist plan state to the current session context.

        ``SessionContext`` is a frozen model — we must never mutate nested
        dicts in place. We always:
          1. shallow-copy the ``workflow_plans`` mapping,
          2. write the new entry into the copy,
          3. wrap it in a fresh ``memory`` dict,
          4. ``model_copy`` the context to derive a new snapshot.
        """
        ctx = current_context()
        if ctx is None or self.harness is None:
            return
        plans_memory = dict(ctx.memory.get("workflow_plans", {}))
        plans_memory[str(plan.plan_id)] = plan.model_dump(mode="json")
        new_memory = {**ctx.memory, "workflow_plans": plans_memory}
        new_ctx = ctx.model_copy(update={"memory": new_memory})
        token = set_current(new_ctx)
        try:
            await _settle(
                self.harness.bus.pm.hook.save_session(session=new_ctx)
            )
        finally:
            reset_current(token)

    @hookimpl
    async def on_step_update(self, plan_id: str, step: WorkflowStep) -> None:
        """Update plan state in memory when a step changes."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return
        updated_steps = tuple(
            step if s.id == step.id else s for s in plan.steps
        )
        plan = plan.model_copy(update={"steps": updated_steps})
        self._plans[plan_id] = plan

    @hookimpl
    async def on_plan_completed(self, plan: WorkflowPlan) -> None:
        """Update plan state and persist to session storage."""
        self._plans[str(plan.plan_id)] = plan
        await self._persist_plan(plan)

    def _emit_context(self) -> SessionContext:
        """Return the current session context for event payloads.

        Falls back to a fresh empty ``SessionContext`` when no session is
        active so the broadcast layer is never asked to emit without one.
        """
        return current_context() or SessionContext()

    async def _broadcast_step_event(self, plan: WorkflowPlan, step: WorkflowStep, event: str) -> None:
        """Broadcast a step-level event via the event bus."""
        try:
            await self.harness.bus.aemit(
                event,
                context=self._emit_context(),
                plan_id=str(plan.plan_id),
                plan_goal=plan.task,
                step_id=step.id,
                step_title=step.title,
                step_status=step.status.value,
                plan_progress=plan.progress,
                use_subagent=step.use_subagent,
            )
        except Exception:
            pass

    async def _broadcast_plan_event(self, plan: WorkflowPlan, event: str) -> None:
        """Broadcast a plan-level event via the event bus."""
        try:
            await self.harness.bus.aemit(
                event,
                context=self._emit_context(),
                plan_id=str(plan.plan_id),
                plan_goal=plan.task,
                plan_status=plan.status,
                step_count=len(plan.steps),
                progress=plan.progress,
                steps=[s.model_dump(mode="json") for s in plan.steps],
            )
        except Exception:
            pass

    @hookimpl
    async def save_plan(self, plan: WorkflowPlan, session_id: str) -> None:
        """Persist plan to session store via hook."""
        await self._persist_plan(plan)

    @hookimpl
    async def load_plan(self, plan_id: str) -> WorkflowPlan | None:
        """Load plan from in-memory cache."""
        return self._plans.get(plan_id)

    @hookimpl
    async def list_plans(self, session_id: str) -> list[WorkflowPlan]:
        """List all cached plans."""
        return list(self._plans.values())

    async def resume_plan(self, plan_id: str) -> WorkflowPlan | None:
        """Resume execution of a persisted plan."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return None
        remaining = [s for s in plan.steps if s.status in (StepStatus.PENDING, StepStatus.FAILED)]
        if not remaining:
            return plan
        return plan

    def _workflow_tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name="workflow_execute",
            description=(
                "为复杂任务生成结构化执行计划并逐步执行。"
                "首先规划一个 To-Do List，然后按依赖顺序执行每个步骤，"
                "追踪进度并在失败时自动重试。"
                "参数 task 是任务描述（必填），model 是使用的模型名称（可选）。"
            ),
            parameters=(
                ToolArg(
                    name="task",
                    type="string",
                    description="复杂任务的描述。系统将为此任务生成并执行一个结构化计划。",
                    required=True,
                ),
                ToolArg(
                    name="model",
                    type="string",
                    description="用于规划和执行的模型名称（默认: default）。",
                    required=False,
                ),
            ),
            timeout_seconds=600.0,
        )

    def _get_plan_status_spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_plan_status",
            description="查看当前执行计划的进度和每个步骤的状态。",
            parameters=(
                ToolArg(
                    name="plan_id",
                    type="string",
                    description="要查询的计划 ID（由 workflow_execute 返回）。",
                    required=True,
                ),
            ),
            timeout_seconds=30.0,
        )

    def _update_plan_spec(self) -> ToolSpec:
        return ToolSpec(
            name="update_plan",
            description=(
                "动态修改正在执行的计划。"
                "支持 4 种操作：add_step（插入新步骤）、skip_step（跳过步骤）、"
                "update_step（修改步骤描述）、cancel_plan（取消计划）。"
                "只能修改尚未执行的步骤，已完成或正在执行的步骤不可修改。"
            ),
            parameters=(
                ToolArg(
                    name="plan_id",
                    type="string",
                    description="要修改的计划 ID。",
                    required=True,
                ),
                ToolArg(
                    name="action",
                    type="string",
                    description="操作类型：add_step / skip_step / update_step / cancel_plan。",
                    required=True,
                ),
                ToolArg(
                    name="step_id",
                    type="string",
                    description="目标步骤 ID（cancel_plan 时不需要）。",
                    required=False,
                ),
                ToolArg(
                    name="new_title",
                    type="string",
                    description="新标题（add_step / update_step 时使用）。",
                    required=False,
                ),
                ToolArg(
                    name="new_description",
                    type="string",
                    description="新描述（add_step / update_step 时使用）。",
                    required=False,
                ),
                ToolArg(
                    name="insert_after",
                    type="string",
                    description="在此步骤之后插入（add_step 时使用）。",
                    required=False,
                ),
                ToolArg(
                    name="reason",
                    type="string",
                    description="修改原因（记录到日志）。",
                    required=False,
                ),
            ),
            timeout_seconds=30.0,
        )

    @hookimpl
    def get_tool_specs(self, context: SessionContext) -> tuple[ToolSpec, ...]:
        return (self._workflow_tool_spec(), self._get_plan_status_spec(), self._update_plan_spec())

    @hookimpl
    async def execute_tool(
        self, context: SessionContext, tool: ToolSpec, arguments: dict[str, object]
    ) -> ToolResult | None:
        if tool.name == "workflow_execute":
            return await self._execute_workflow(context, tool, arguments)
        if tool.name == "get_plan_status":
            return await self._get_plan_status(context, tool, arguments)
        if tool.name == "update_plan":
            return await self._handle_update_plan(arguments)
        return None

    async def _execute_workflow(
        self, context: SessionContext, tool: ToolSpec, arguments: dict[str, object]
    ) -> ToolResult:
        task = arguments.get("task")
        if not task or not isinstance(task, str):
            return ToolResult(
                tool_name="workflow_execute",
                status=ToolResultStatus.ERROR,
                error="'task' 参数必须是字符串。",
                output={},
            )

        model = arguments.get("model", "default")
        if not isinstance(model, str):
            model = "default"

        providers = await _settle(
            self.harness.bus.pm.hook.get_llm_providers(context=context)
        )
        available_models: list[str] = []
        for value in providers:
            if value is not None:
                available_models.extend(value if isinstance(value, tuple) else (value,))
        if model not in available_models and available_models:
            model = available_models[0]

        if self.harness is None:
            return ToolResult(
                tool_name="workflow_execute",
                status=ToolResultStatus.ERROR,
                error="Harness 未初始化，无法执行工作流。",
                output={},
            )

        try:
            plan = await self._generate_plan(context, task, model)
            logger.info("[workflow] plan parsed: id=%s task=%s steps=%d", plan.plan_id, plan.task, len(plan.steps))
            result = await self._execute_plan(context, plan, model)
            logger.info("[workflow] plan finished: id=%s status=%s", plan.plan_id, result.get("status"))
            return ToolResult(
                tool_name="workflow_execute",
                status=ToolResultStatus.OK,
                output=result,
            )
        except Exception as exc:
            logger.exception("workflow_execute failed")
            return ToolResult(
                tool_name="workflow_execute",
                status=ToolResultStatus.ERROR,
                error=f"工作流执行失败: {exc}",
                output={},
            )

    async def _get_plan_status(
        self, context: SessionContext, tool: ToolSpec, arguments: dict[str, object]
    ) -> ToolResult:
        plan_id = arguments.get("plan_id")
        if not plan_id or not isinstance(plan_id, str):
            return ToolResult(
                tool_name="get_plan_status",
                status=ToolResultStatus.ERROR,
                error="'plan_id' 参数必须是字符串。",
                output={},
            )

        plan = self._plans.get(plan_id)
        if plan is None:
            return ToolResult(
                tool_name="get_plan_status",
                status=ToolResultStatus.ERROR,
                error=f"未找到 plan_id={plan_id} 的计划。",
                output={},
            )

        return ToolResult(
            tool_name="get_plan_status",
            status=ToolResultStatus.OK,
            output={
                "plan_id": str(plan.plan_id),
                "task": plan.task,
                "status": plan.status,
                "steps": [s.model_dump(mode="json") for s in plan.steps],
            },
        )

    async def _handle_update_plan(self, arguments: dict[str, object]) -> ToolResult:
        """Handle the ``update_plan`` tool with full validation."""
        plan_id = arguments.get("plan_id")
        if not plan_id or not isinstance(plan_id, str):
            return ToolResult(
                tool_name="update_plan",
                status=ToolResultStatus.ERROR,
                error="'plan_id' 参数必须是字符串。",
                output={},
            )

        action = arguments.get("action")
        if not isinstance(action, str):
            return ToolResult(
                tool_name="update_plan",
                status=ToolResultStatus.ERROR,
                error="'action' 参数必须是字符串。",
                output={},
            )
        try:
            plan_action = PlanAction(action)
        except ValueError:
            return ToolResult(
                tool_name="update_plan",
                status=ToolResultStatus.ERROR,
                error=f"无效的 action: {action}。允许的值: add_step, skip_step, update_step, cancel_plan。",
                output={},
            )

        plan = self._plans.get(plan_id)
        if plan is None:
            return ToolResult(
                tool_name="update_plan",
                status=ToolResultStatus.ERROR,
                error=f"未找到 plan_id={plan_id} 的计划。",
                output={},
            )

        if plan.status in ("completed", "failed", "cancelled"):
            return ToolResult(
                tool_name="update_plan",
                status=ToolResultStatus.ERROR,
                error=f"计划已处于终态 ({plan.status})，不能再修改。",
                output={},
            )

        reason = arguments.get("reason")
        reason_str = str(reason) if reason is not None else None
        if reason_str:
            logger.info("update_plan plan_id=%s action=%s reason=%s", plan_id, plan_action.value, reason_str)

        updated_plan = await self._apply_plan_action(plan, plan_action, arguments)
        if updated_plan is None:
            return ToolResult(
                tool_name="update_plan",
                status=ToolResultStatus.ERROR,
                error="更新计划失败，请检查参数。",
                output={},
            )

        self._plans[plan_id] = updated_plan
        if self.harness is not None:
            await _settle(
                self.harness.bus.pm.hook.on_plan_update(plan=updated_plan)
            )
        await self._persist_plan(updated_plan)
        return ToolResult(
            tool_name="update_plan",
            status=ToolResultStatus.OK,
            output={"plan": updated_plan.model_dump(mode="json")},
        )

    async def _apply_plan_action(
        self, plan: WorkflowPlan, action: PlanAction, arguments: dict[str, object]
    ) -> WorkflowPlan | None:
        """Apply a mutation to the plan, returning a new frozen plan or None on error."""
        match action:
            case PlanAction.CANCEL_PLAN:
                cancelled_steps = tuple(
                    s.model_copy(update={"status": StepStatus.SKIPPED, "error": "Plan cancelled"})
                    if s.status == StepStatus.PENDING
                    else s
                    for s in plan.steps
                )
                return plan.model_copy(update={"status": "cancelled", "steps": cancelled_steps})

            case PlanAction.SKIP_STEP:
                step_id = arguments.get("step_id")
                if not step_id or not isinstance(step_id, str):
                    return None
                target = next((s for s in plan.steps if s.id == step_id), None)
                if target is None:
                    return None
                if target.status != StepStatus.PENDING:
                    return None
                updated_steps = tuple(
                    s.model_copy(update={"status": StepStatus.SKIPPED, "error": "Skipped by user"})
                    if s.id == step_id and s.status == StepStatus.PENDING
                    else s
                    for s in plan.steps
                )
                return plan.model_copy(update={"steps": updated_steps})

            case PlanAction.UPDATE_STEP:
                step_id = arguments.get("step_id")
                new_title = arguments.get("new_title")
                new_description = arguments.get("new_description")
                if not step_id or not isinstance(step_id, str):
                    return None
                target = next((s for s in plan.steps if s.id == step_id), None)
                if target is None:
                    return None
                if target.status != StepStatus.PENDING:
                    return None
                updates: dict[str, Any] = {}
                if new_title is not None and isinstance(new_title, str):
                    updates["title"] = new_title
                if new_description is not None and isinstance(new_description, str):
                    updates["description"] = new_description
                if not updates:
                    return None
                updated_steps = tuple(
                    s.model_copy(update=updates) if s.id == step_id else s
                    for s in plan.steps
                )
                return plan.model_copy(update={"steps": updated_steps})

            case PlanAction.ADD_STEP:
                step_id = arguments.get("step_id")
                new_title = arguments.get("new_title")
                new_description = arguments.get("new_description")
                insert_after = arguments.get("insert_after")
                if not step_id or not isinstance(step_id, str):
                    return None
                if any(s.id == step_id for s in plan.steps):
                    return None
                if not new_description or not isinstance(new_description, str):
                    return None
                title = str(new_title) if new_title is not None else step_id
                new_step = WorkflowStep(
                    id=step_id,
                    title=title,
                    description=new_description,
                    depends_on=[insert_after] if insert_after and isinstance(insert_after, str) else [],
                    status=StepStatus.PENDING,
                )
                if insert_after and isinstance(insert_after, str):
                    idx = next((i for i, s in enumerate(plan.steps) if s.id == insert_after), None)
                    if idx is None:
                        return None
                    updated_steps = list(plan.steps)
                    updated_steps.insert(idx + 1, new_step)
                    return plan.model_copy(update={"steps": tuple(updated_steps)})
                return plan.model_copy(update={"steps": (*plan.steps, new_step)})

        return None

    async def _generate_plan(self, context: SessionContext, task: str, model: str) -> WorkflowPlan:
        """Ask the LLM to produce a structured JSON plan for the task."""
        prompt = (
            "Generate a structured execution plan for the following task.\n"
            "Return ONLY a JSON array of steps. Each step must have:\n"
            "- id: unique string identifier (e.g. \"step-1\")\n"
            "- description: what to do in this step (1-2 sentences)\n"
            "- depends_on: array of step ids this step depends on (empty if none)\n\n"
            "Example:\n"
            "[\n"
            "  {\"id\": \"step-1\", \"description\": \"Research the topic\", \"depends_on\": []},\n"
            "  {\"id\": \"step-2\", \"description\": \"Write a summary\", \"depends_on\": [\"step-1\"]},\n"
            "  {\"id\": \"step-3\", \"description\": \"Review and finalize\", \"depends_on\": [\"step-2\"]}\n"
            "]\n\n"
            f"Task: {task}\n\n"
            "JSON:"
        )

        request = LLMRequest(
            model=model,
            messages=(Message(role=Role.USER, content=prompt),),
        )

        resp_content = ""
        async for chunk in self.harness.stream_session(
            AgentConfig(name="workflow-plan", model=model),
            initial_text=prompt,
        ):
            resp_content += chunk.delta

        if not resp_content.strip():
            raise RuntimeError("No LLM provider available for plan generation")

        plan_data = self._extract_json(resp_content)
        if not isinstance(plan_data, list):
            raise RuntimeError(f"Expected JSON array from LLM, got: {resp_content[:200]}")

        steps: list[WorkflowStep] = []
        for step_data in plan_data:
            if not isinstance(step_data, dict):
                continue
            step = WorkflowStep(
                id=step_data.get("id", f"step-{len(steps)+1}"),
                title=step_data.get("title", step_data.get("id", f"step-{len(steps)+1}")),
                description=step_data.get("description", ""),
                depends_on=step_data.get("depends_on", []),
            )
            steps.append(step)

        plan = WorkflowPlan(task=task, steps=tuple(steps))
        await _settle(
            self.harness.bus.pm.hook.on_plan_created(plan=plan)
        )
        await self._broadcast_plan_event(plan, "plan_created")
        logger.info("[workflow] plan parsed: id=%s task=%s steps=%d", plan.plan_id, plan.task, len(plan.steps))
        return plan

    @staticmethod
    def _validate_no_cycles(plan: WorkflowPlan) -> None:
        """Raise ``ValueError`` if the plan's dependency graph contains a cycle."""
        step_ids = {s.id for s in plan.steps}
        adjacency: dict[str, list[str]] = {s.id: list(s.depends_on) for s in plan.steps}
        visited: set[str] = set()
        path: set[str] = set()

        def dfs(node: str) -> None:
            if node in path:
                cycle = " -> ".join(sorted({*path, node}))
                raise ValueError(f"循环依赖检测失败: {cycle}")
            if node in visited:
                return
            path.add(node)
            for neighbour in adjacency.get(node, []):
                if neighbour in step_ids:
                    dfs(neighbour)
            path.remove(node)
            visited.add(node)

        for step in plan.steps:
            dfs(step.id)

    async def _execute_plan(self, context: SessionContext, plan: WorkflowPlan, model: str) -> dict[str, Any]:
        """Execute the plan step by step, respecting dependencies and retries."""
        self._validate_no_cycles(plan)
        results: dict[str, WorkflowStep] = {}
        remaining = [s for s in plan.steps if s.status not in (StepStatus.COMPLETED, StepStatus.SKIPPED)]
        original_status = {s.id: s.status for s in plan.steps}
        original_steps = {s.id: s for s in plan.steps}

        logger.info("[workflow] plan=%s start executing %d steps", plan.plan_id, len(plan.steps))
        await self._broadcast_plan_event(plan, "plan_created")

        while remaining:
            # Cascade-fail: any remaining step whose dep has FAILED must be
            # marked SKIPPED with an explicit reason so it never lingers as
            # "remaining" forever.
            def _step_status(dep: str):
                if dep in results:
                    return results[dep].status
                return original_steps.get(dep, None).status if dep in original_steps else None

            for s in remaining:
                failed_dep = next(
                    (dep for dep in s.depends_on if _step_status(dep) == StepStatus.FAILED),
                    None,
                )
                if failed_dep is not None and s.status == StepStatus.PENDING:
                    s = s.model_copy(
                        update={
                            "status": StepStatus.SKIPPED,
                            "error": f"依赖步骤 {failed_dep} 失败，已自动跳过",
                        }
                    )
                    results[s.id] = s
                    await _settle(
                        self.harness.bus.pm.hook.on_step_update(
                            plan_id=str(plan.plan_id), step=s
                        )
                    )
                    await self._broadcast_step_event(plan, s, "plan_step_complete")
            remaining = [s for s in remaining if s.id not in results]

            def _dep_satisfied(dep: str) -> bool:
                st = _step_status(dep)
                return st in (StepStatus.COMPLETED, StepStatus.SKIPPED)

            ready = [s for s in remaining if all(_dep_satisfied(dep) for dep in s.depends_on)]

            if not ready:
                for s in remaining:
                    if s.status == StepStatus.PENDING:
                        s = s.model_copy(update={"status": StepStatus.SKIPPED, "error": "Dependencies not satisfied"})
                        results[s.id] = s
                        await _settle(
                            self.harness.bus.pm.hook.on_step_update(plan_id=str(plan.plan_id), step=s)
                        )
                        await self._broadcast_step_event(plan, s, "plan_step_complete")
                break

            remaining = [s for s in remaining if s not in ready]
            logger.info("[workflow] plan=%s ready batch: %s", plan.plan_id, [s.id for s in ready])

            running_batch: list[WorkflowStep] = []
            for step in ready:
                step = step.model_copy(update={"status": StepStatus.RUNNING, "started_at": _utc_now()})
                running_batch.append(step)
                await _settle(
                    self.harness.bus.pm.hook.on_step_update(plan_id=str(plan.plan_id), step=step)
                )
                await self._broadcast_step_event(plan, step, "plan_step_start")
                logger.info("[workflow] plan=%s step=%s -> running", plan.plan_id, step.id)

            batch_results: dict[str, WorkflowStep] = {}
            async with asyncio.TaskGroup() as tg:
                for step in running_batch:
                    tg.create_task(self._execute_single_step(context, plan, step, model, batch_results))

            results.update(batch_results)
            updated_plan = plan.model_copy(update={"steps": tuple(results.values())})
            progress = sum(1 for s in results.values() if s.status == StepStatus.COMPLETED) / max(len(plan.steps), 1)
            updated_plan = updated_plan.model_copy(update={"progress": progress})
            plan = updated_plan
            await self._persist_plan(plan)
            remaining = [s for s in remaining if s.id not in batch_results]

        if all(s.status == StepStatus.COMPLETED for s in results.values()):
            plan_status = "completed"
        elif any(s.status == StepStatus.FAILED for s in results.values()):
            plan_status = "failed"
        else:
            plan_status = "running"

        completed_plan = plan.model_copy(update={"status": plan_status})
        self._plans[str(completed_plan.plan_id)] = completed_plan
        await _settle(
            self.harness.bus.pm.hook.on_plan_completed(plan=completed_plan)
        )
        await self._broadcast_plan_event(completed_plan, "plan_completed")
        await self._persist_plan(completed_plan)
        logger.info("[workflow] plan=%s finished status=%s", completed_plan.plan_id, plan_status)

        all_steps: dict[str, WorkflowStep] = {**original_steps, **results}
        return {
            "plan_id": str(completed_plan.plan_id),
            "task": plan.task,
            "status": plan_status,
            "steps": [s.model_dump(mode="json") for s in all_steps.values()],
        }

    async def _execute_single_step(
        self,
        context: SessionContext,
        plan: WorkflowPlan,
        step: WorkflowStep,
        model: str,
        results: dict[str, WorkflowStep],
    ) -> None:
        """Execute a single step with retry logic, storing the result in ``results``."""
        start_time = time.monotonic()
        max_retries = step.max_retries
        for attempt in range(max_retries + 1):
            if attempt > 0:
                step = step.model_copy(update={"status": StepStatus.RETRYING, "retries": attempt})
                await _settle(
                    self.harness.bus.pm.hook.on_step_update(plan_id=str(plan.plan_id), step=step)
                )
                await self._broadcast_step_event(plan, step, "plan_step_retry")

            try:
                output, error, status = await asyncio.wait_for(
                    self._run_step_isolated(context, plan, step, model),
                    timeout=120.0,
                )
                if status == StepStatus.COMPLETED:
                    step = step.model_copy(update={
                        "result": output or "",
                        "status": StepStatus.COMPLETED,
                        "error": None,
                    })
                else:
                    step = step.model_copy(update={
                        "error": error or "step failed",
                        "status": StepStatus.FAILED,
                    })

                duration = time.monotonic() - start_time
                step = step.model_copy(update={"duration_seconds": round(duration, 2)})
                await _settle(
                    self.harness.bus.pm.hook.on_step_update(plan_id=str(plan.plan_id), step=step)
                )
                await self._broadcast_step_event(plan, step, "plan_step_complete")
                break
            except asyncio.TimeoutError:
                duration = time.monotonic() - start_time
                step = step.model_copy(update={
                    "error": f"步骤超时（120s）",
                    "status": StepStatus.FAILED,
                    "duration_seconds": round(duration, 2),
                })
                await _settle(
                    self.harness.bus.pm.hook.on_step_update(plan_id=str(plan.plan_id), step=step)
                )
                await self._broadcast_step_event(plan, step, "plan_step_complete")
                logger.warning("[workflow] plan=%s step=%s TIMEOUT after 120s", plan.plan_id, step.id)
                break
            except Exception as exc:
                if attempt < max_retries:
                    continue
                duration = time.monotonic() - start_time
                step = step.model_copy(update={"error": str(exc), "status": StepStatus.FAILED, "duration_seconds": round(duration, 2)})
                await _settle(
                    self.harness.bus.pm.hook.on_step_update(plan_id=str(plan.plan_id), step=step)
                )
                await self._broadcast_step_event(plan, step, "plan_step_complete")
                logger.exception("[workflow] plan=%s step=%s ERROR", plan.plan_id, step.id)

        step = step.model_copy(update={"finished_at": _utc_now()})
        results[step.id] = step

    async def _run_step_isolated(
        self,
        parent_ctx: SessionContext,
        plan: WorkflowPlan,
        step: WorkflowStep,
        model: str,
    ) -> tuple[str | None, str | None, StepStatus]:
        """Run a single workflow step inside a brand-new session scope.

        ``open_session`` resets the task-scoped ``ContextVar`` so nested
        ``run_session`` / ``spawn_subagent`` calls cannot re-enter the outer
        session's asyncio locks. This prevents the deadlock observed when a
        workflow step tried to acquire the parent session's lock.
        """
        from pyharness.context import open_session

        logger.info("[workflow] plan=%s step=%s -> isolated run started", plan.plan_id, step.id)
        async with open_session(namespace=f"workflow-step:{step.id}") as step_ctx:
            if step.use_subagent:
                spec = SubagentSpec(
                    name=step.id,
                    task=step.description,
                    model=model,
                    max_turns=step.subagent_max_turns,
                    timeout=step.subagent_timeout,
                )
                result = await self.harness.spawn_subagent(
                    spec,
                    parent_tools=list(
                        s
                        for specs in self.harness.bus.pm.hook.get_tool_specs(context=parent_ctx)
                        if specs
                        for s in specs
                    ),
                    parent_config=AgentConfig(name="workflow", model=model),
                )
                ok = result.status == "ok"
                logger.info("[workflow] plan=%s step=%s subagent done status=%s", plan.plan_id, step.id, result.status)
                return (
                    result.output or "",
                    result.error,
                    StepStatus.COMPLETED if ok else StepStatus.FAILED,
                )

            session = await self.harness.run_session(
                AgentConfig(name=step.id, model=model, max_steps=5),
                initial_text=step.description,
                namespace=f"workflow-step:{step.id}",
            )
            last = session.messages[-1] if session.messages else None
            logger.info("[workflow] plan=%s step=%s run_session done", plan.plan_id, step.id)
            return ((last.content if last else "") or "", None, StepStatus.COMPLETED)

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response text."""
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise RuntimeError(f"Could not extract JSON from LLM response: {text[:200]}")


__all__ = ["WorkflowPlugin"]
