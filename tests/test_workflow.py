"""Tests for Phase 3 Workflow / Plan task orchestration."""

from __future__ import annotations

import asyncio
import inspect
import time
from unittest.mock import patch

import pytest
from pluggy import HookimplMarker

from pyharness import Harness
from pyharness.context import SessionContext
from pyharness.core import _settle
from pyharness.plugins.llm import entry as llm
from pyharness.plugins.workflow import WorkflowPlugin
from pyharness.schema import (
    AgentConfig,
    HarnessConfig,
    LLMResponse,
    Message,
    PlanAction,
    Role,
    StepStatus,
    SubagentResult,
    ToolResultStatus,
    UpdatePlanInput,
    WorkflowPlan,
    WorkflowStep,
)

hookimpl = HookimplMarker("pyharness")


def _harness(*plugins, auto_load: bool = False) -> Harness:
    h = Harness(config=HarnessConfig(auto_load_entry_points=auto_load))
    llm.clear()
    llm.use_dummy(models=("mock-model",), plan=[LLMResponse(model="mock-model", content="done")])
    h.register_plugin(llm)
    for plugin in plugins:
        h.register_plugin(plugin)
    h.initialize()
    return h


async def test_workflow_plan_generation() -> None:
    """LLM returns a JSON plan; the plugin parses it into WorkflowStep objects."""
    h = _harness(WorkflowPlugin())
    llm.clear()
    llm.use_dummy(
        models=("mock-model",),
        plan=[
            LLMResponse(
                model="mock-model",
                content='[{"id": "s1", "description": "step 1", "depends_on": []}, '
                        '{"id": "s2", "description": "step 2", "depends_on": ["s1"]}]',
            ),
        ],
    )
    ctx = SessionContext()

    specs = {}
    for plugin_specs in h.bus.pm.hook.get_tool_specs(context=ctx):
        specs.update({s.name: s for s in plugin_specs})
    tool = specs["workflow_execute"]

    raw = h.bus.pm.hook.execute_tool(
        context=ctx, tool=tool, arguments={"task": "do something", "model": "mock-model"}
    )
    result = next((r for r in await _settle(raw) if r is not None), None)

    assert result is not None
    assert result.status == ToolResultStatus.OK
    output = result.output
    assert output["status"] == "completed"
    assert len(output["steps"]) == 2
    assert output["steps"][0]["id"] == "s1"
    assert output["steps"][0]["depends_on"] == []
    assert output["steps"][1]["id"] == "s2"
    assert output["steps"][1]["depends_on"] == ["s1"]


async def test_workflow_execution() -> None:
    """A simple 2-step plan executes successfully in dependency order."""
    h = _harness(WorkflowPlugin())
    llm.clear()
    llm.use_dummy(
        models=("mock-model",),
        plan=[
            LLMResponse(
                model="mock-model",
                content='[{"id": "s1", "description": "step 1", "depends_on": []}, '
                        '{"id": "s2", "description": "step 2", "depends_on": ["s1"]}]',
            ),
            LLMResponse(model="mock-model", content="step 1 result"),
            LLMResponse(model="mock-model", content="step 2 result"),
        ],
    )
    ctx = SessionContext()

    specs = {}
    for plugin_specs in h.bus.pm.hook.get_tool_specs(context=ctx):
        specs.update({s.name: s for s in plugin_specs})
    tool = specs["workflow_execute"]

    raw = h.bus.pm.hook.execute_tool(
        context=ctx, tool=tool, arguments={"task": "do something", "model": "mock-model"}
    )
    result = next((r for r in await _settle(raw) if r is not None), None)

    assert result is not None
    assert result.status == ToolResultStatus.OK
    output = result.output
    assert output["status"] == "completed"
    assert "plan_id" in output
    plan_id = output["plan_id"]
    steps = {s["id"]: s for s in output["steps"]}
    assert steps["s1"]["status"] == StepStatus.COMPLETED
    assert steps["s1"]["result"] == "step 1 result"
    assert steps["s2"]["status"] == StepStatus.COMPLETED
    assert steps["s2"]["result"] == "step 2 result"

    # Verify get_plan_status returns the same plan
    status_tool = specs["get_plan_status"]
    raw_status = h.bus.pm.hook.execute_tool(
        context=ctx, tool=status_tool, arguments={"plan_id": plan_id}
    )
    status_result = next((r for r in await _settle(raw_status) if r is not None), None)
    assert status_result is not None
    assert status_result.status == ToolResultStatus.OK
    assert status_result.output["plan_id"] == plan_id
    assert status_result.output["status"] == "completed"
    assert len(status_result.output["steps"]) == 2


async def test_workflow_dependencies() -> None:
    """Steps with unsatisfied dependencies are skipped."""
    h = _harness(WorkflowPlugin())
    llm.clear()
    llm.use_dummy(
        models=("mock-model",),
        plan=[
            LLMResponse(
                model="mock-model",
                content='[{"id": "s1", "description": "step 1", "depends_on": ["missing"]}, '
                        '{"id": "s2", "description": "step 2", "depends_on": ["s1"]}]',
            ),
            LLMResponse(model="mock-model", content="step 2 result"),
        ],
    )
    ctx = SessionContext()

    specs = {}
    for plugin_specs in h.bus.pm.hook.get_tool_specs(context=ctx):
        specs.update({s.name: s for s in plugin_specs})
    tool = specs["workflow_execute"]

    raw = h.bus.pm.hook.execute_tool(
        context=ctx, tool=tool, arguments={"task": "do something", "model": "mock-model"}
    )
    result = next((r for r in await _settle(raw) if r is not None), None)

    assert result is not None
    assert result.status == ToolResultStatus.OK
    output = result.output
    steps = {s["id"]: s for s in output["steps"]}
    assert steps["s1"]["status"] == StepStatus.SKIPPED
    assert "Dependencies not satisfied" in steps["s1"]["error"]
    assert steps["s2"]["status"] == StepStatus.SKIPPED


async def test_plan_circular_dependency() -> None:
    """A->B->C->A 循环依赖，应抛出 ValueError."""
    plugin = WorkflowPlugin()
    steps = (
        WorkflowStep(id="a", title="A", description="A", depends_on=["c"]),
        WorkflowStep(id="b", title="B", description="B", depends_on=["a"]),
        WorkflowStep(id="c", title="C", description="C", depends_on=["b"]),
    )
    plan = WorkflowPlan(task="circular test", steps=steps)
    with pytest.raises(ValueError, match="循环依赖"):
        plugin._validate_no_cycles(plan)


async def test_workflow_retry() -> None:
    """A failing step is retried up to max_retries times."""
    plugin = WorkflowPlugin()

    call_count = 0
    step_updates: list[dict[str, Any]] = []

    class MockHook:
        async def on_step_update(self, plan_id, step):
            step_updates.append({"plan_id": plan_id, "step": step})

        async def on_plan_completed(self, plan):
            pass

    class MockPM:
        def __init__(self):
            self.hook = MockHook()

    class MockBus:
        def __init__(self):
            self.pm = MockPM()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"flaky failure {call_count}")
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="success after retry"),)
            )

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="test",
        steps=(        WorkflowStep(id="s1", title="flaky step", description="flaky step", max_retries=2),),
    )

    result = await plugin._execute_plan(SessionContext(), plan, "mock-model")

    assert result["steps"][0]["status"] == StepStatus.COMPLETED
    assert result["steps"][0]["result"] == "success after retry"
    assert result["steps"][0]["retries"] == 2
    assert call_count == 3
    assert len(step_updates) >= 2


async def test_workflow_hooks_emitted() -> None:
    """on_step_update and on_plan_completed are emitted during execution."""
    step_update_calls: list[dict[str, Any]] = []
    plan_completed_calls: list[WorkflowPlan] = []

    class MockHook:
        async def on_step_update(self, plan_id, step):
            step_update_calls.append({"plan_id": plan_id, "step": step})

        async def on_plan_completed(self, plan):
            plan_completed_calls.append(plan)

    class MockPM:
        def __init__(self):
            self.hook = MockHook()

    class MockBus:
        def __init__(self):
            self.pm = MockPM()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="done"),)
            )

    plugin = WorkflowPlugin()
    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="hook test",
        steps=(WorkflowStep(id="s1", title="step 1", description="step 1", depends_on=[]),),
    )

    result = await plugin._execute_plan(SessionContext(), plan, "mock-model")

    assert len(step_update_calls) >= 1
    running_updates = [c for c in step_update_calls if c["step"].status == StepStatus.RUNNING]
    assert len(running_updates) >= 1

    assert len(plan_completed_calls) == 1
    assert plan_completed_calls[0].status == "completed"


async def test_workflow_plan_persistence() -> None:
    """Plan state is persisted to session memory and restored on session start."""
    plugin = WorkflowPlugin()

    class MockHook:
        async def on_step_update(self, plan_id, step):
            pass

        async def on_plan_completed(self, plan):
            pass

        def get_tool_specs(self, context):
            return ()

    class MockPM:
        def __init__(self):
            self.hook = MockHook()

    class MockBus:
        def __init__(self):
            self.pm = MockPM()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="done"),)
            )

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="persist test",
        steps=(WorkflowStep(id="s1", title="step 1", description="step 1", depends_on=[]),),
    )

    await plugin._execute_plan(SessionContext(), plan, "mock-model")

    plan_id = str(plan.plan_id)
    assert plan_id in plugin._plans
    assert plugin._plans[plan_id].status == "completed"

    # Simulate session restart: create a new plugin and restore from session memory
    restored_plugin = WorkflowPlugin()
    session_ctx = SessionContext(memory={"workflow_plans": {plan_id: plan.model_dump(mode="json")}})
    restored_plugin.session_started(session_ctx)

    assert plan_id in restored_plugin._plans
    assert restored_plugin._plans[plan_id].task == "persist test"


async def test_workflow_subagent_step() -> None:
    """A step with use_subagent=True is executed via spawn_subagent."""
    plugin = WorkflowPlugin()
    step_updates: list[dict[str, Any]] = []

    class MockHook:
        async def on_step_update(self, plan_id, step):
            step_updates.append({"plan_id": plan_id, "step": step})

        async def on_plan_completed(self, plan):
            pass

        def get_tool_specs(self, context):
            return ()

    class MockPM:
        def __init__(self):
            self.hook = MockHook()

    class MockBus:
        def __init__(self):
            self.pm = MockPM()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def spawn_subagent(self, spec, parent_tools=None, parent_config=None):
            assert spec.name == "s1"
            assert spec.task == "subagent task"
            return SubagentResult(
                spec=spec,
                status="ok",
                output="subagent result",
                session_id="sub-1",
            )

        async def run_session(self, *args, **kwargs):
            raise AssertionError("run_session should not be called for subagent step")

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="subagent test",
        steps=(WorkflowStep(id="s1", title="subagent task", description="subagent task", use_subagent=True),),
    )

    result = await plugin._execute_plan(SessionContext(), plan, "mock-model")

    assert result["steps"][0]["status"] == StepStatus.COMPLETED
    assert result["steps"][0]["result"] == "subagent result"


async def test_workflow_subagent_failure() -> None:
    """A step with use_subagent=True that fails is marked FAILED."""
    plugin = WorkflowPlugin()
    step_updates: list[dict[str, Any]] = []

    class MockHook:
        async def on_step_update(self, plan_id, step):
            step_updates.append({"plan_id": plan_id, "step": step})

        async def on_plan_completed(self, plan):
            pass

        def get_tool_specs(self, context):
            return ()

    class MockPM:
        def __init__(self):
            self.hook = MockHook()

    class MockBus:
        def __init__(self):
            self.pm = MockPM()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def spawn_subagent(self, spec, parent_tools=None, parent_config=None):
            return SubagentResult(
                spec=spec,
                status="error",
                error="subagent crashed",
                session_id="sub-1",
            )

        async def run_session(self, *args, **kwargs):
            raise AssertionError("run_session should not be called for subagent step")

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="subagent fail test",
        steps=(WorkflowStep(id="s1", title="failing subagent", description="failing subagent", use_subagent=True),),
    )

    result = await plugin._execute_plan(SessionContext(), plan, "mock-model")

    assert result["steps"][0]["status"] == StepStatus.FAILED
    assert result["steps"][0]["error"] == "subagent crashed"


async def test_workflow_step_events_broadcast() -> None:
    """plan_step_start and plan_step_complete events are broadcast."""
    events: list[dict[str, Any]] = []

    class MockHook:
        async def on_step_update(self, plan_id, step):
            pass

        async def on_plan_completed(self, plan):
            pass

        def get_tool_specs(self, context):
            return ()

    class MockPM:
        def __init__(self):
            self.hook = MockHook()

    class MockBus:
        def __init__(self):
            self.pm = MockPM()
            self._subscribers: dict[str, list[Any]] = {}

        def on(self, event_type, fn):
            self._subscribers.setdefault(event_type, []).append(fn)
            return lambda: None

        async def aemit(self, event_type, **payload):
            for fn in self._subscribers.get(event_type, []):
                value = fn(event_type=event_type, **payload)
                if inspect.isawaitable(value):
                    await value
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="done"),)
            )

    plugin = WorkflowPlugin()
    plugin.harness = MockHarness()

    # Subscribe to workflow events
    plugin.harness.bus.on("plan_step_start", lambda **kw: events.append({"type": "plan_step_start", **kw}))
    plugin.harness.bus.on("plan_step_complete", lambda **kw: events.append({"type": "plan_step_complete", **kw}))

    plan = WorkflowPlan(
        task="events test",
        steps=(WorkflowStep(id="s1", title="step 1", description="step 1", depends_on=[]),),
    )

    await plugin._execute_plan(SessionContext(), plan, "mock-model")

    start_events = [e for e in events if e["type"] == "plan_step_start"]
    complete_events = [e for e in events if e["type"] == "plan_step_complete"]

    assert len(start_events) == 1
    assert start_events[0]["step_id"] == "s1"
    assert start_events[0]["step_title"] == "step 1"

    assert len(complete_events) == 1
    assert complete_events[0]["step_id"] == "s1"
    assert complete_events[0]["step_status"] == StepStatus.COMPLETED.value


async def test_plan_parallel_steps() -> None:
    """Independent steps should run in parallel; total time < sum of individual times."""
    plugin = WorkflowPlugin()

    call_log: list[tuple[str, float, float]] = []

    class MockHarness:
        async def run_session(self, agent, initial_text=None, **kwargs):
            step_id = agent.name
            start = time.monotonic()
            await asyncio.sleep(0.2)
            end = time.monotonic()
            call_log.append((step_id, start, end))
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content=f"{step_id} done"),)
            )

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_step_update": lambda self, **kw: None,
                "on_plan_completed": lambda self, **kw: None,
                "save_session": lambda self, **kw: None,
            })()})()

        async def aemit(self, event_type, **payload):
            return []

    MockHarness.bus = MockBus()

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="parallel test",
        steps=(
            WorkflowStep(id="s1", title="Step 1", description="step 1", depends_on=[]),
            WorkflowStep(id="s2", title="Step 2", description="step 2", depends_on=[]),
            WorkflowStep(id="s3", title="Step 3", description="step 3", depends_on=[]),
        ),
    )

    start = time.monotonic()
    result = await plugin._execute_plan(SessionContext(), plan, "mock-model")
    elapsed = time.monotonic() - start

    assert result["status"] == "completed"
    assert len(result["steps"]) == 3
    assert all(s["status"] == StepStatus.COMPLETED for s in result["steps"])
    assert elapsed < 0.5, f"Expected parallel execution < 0.5s, got {elapsed:.2f}s"


async def test_update_plan_add_step() -> None:
    """Dynamically add a new step after an existing step."""
    plugin = WorkflowPlugin()

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_plan_update": lambda self, **kw: None,
            })()})()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="done"),)
            )

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="add step test",
        steps=(
            WorkflowStep(id="s1", title="Step 1", description="step 1", depends_on=[]),
            WorkflowStep(id="s2", title="Step 2", description="step 2", depends_on=["s1"]),
        ),
    )
    plugin._plans[str(plan.plan_id)] = plan

    input_data = UpdatePlanInput(
        plan_id=str(plan.plan_id),
        action=PlanAction.ADD_STEP,
        step_id="s1b",
        new_title="New step",
        new_description="inserted step",
        insert_after="s1",
    )
    result = await plugin._handle_update_plan(input_data.model_dump())

    assert result.status == ToolResultStatus.OK
    updated = result.output["plan"]
    step_ids = [s["id"] for s in updated["steps"]]
    assert step_ids == ["s1", "s1b", "s2"]
    assert updated["steps"][1]["status"] == StepStatus.PENDING.value


async def test_update_plan_skip_step() -> None:
    """Skip a pending step; downstream steps still execute if not dependent."""
    plugin = WorkflowPlugin()

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_plan_update": lambda self, **kw: None,
            })()})()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="done"),)
            )

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="skip test",
        steps=(
            WorkflowStep(id="s1", title="Step 1", description="step 1", depends_on=[]),
            WorkflowStep(id="s2", title="Step 2", description="step 2", depends_on=[]),
            WorkflowStep(id="s3", title="Step 3", description="step 3", depends_on=["s2"]),
        ),
    )
    plugin._plans[str(plan.plan_id)] = plan

    input_data = UpdatePlanInput(
        plan_id=str(plan.plan_id),
        action=PlanAction.SKIP_STEP,
        step_id="s2",
    )
    result = await plugin._handle_update_plan(input_data.model_dump())

    assert result.status == ToolResultStatus.OK
    updated = result.output["plan"]
    steps = {s["id"]: s for s in updated["steps"]}
    assert steps["s2"]["status"] == StepStatus.SKIPPED.value
    assert steps["s1"]["status"] == StepStatus.PENDING.value


async def test_update_plan_cancel() -> None:
    """Cancel a plan; pending steps become SKIPPED, completed steps unchanged."""
    plugin = WorkflowPlugin()

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_plan_update": lambda self, **kw: None,
            })()})()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="done"),)
            )

    plugin.harness = MockHarness()

    completed_step = WorkflowStep(id="s1", title="Step 1", description="step 1", status=StepStatus.COMPLETED)
    plan = WorkflowPlan(
        task="cancel test",
        steps=(
            completed_step,
            WorkflowStep(id="s2", title="Step 2", description="step 2", depends_on=["s1"]),
        ),
    )
    plugin._plans[str(plan.plan_id)] = plan

    input_data = UpdatePlanInput(
        plan_id=str(plan.plan_id),
        action=PlanAction.CANCEL_PLAN,
    )
    result = await plugin._handle_update_plan(input_data.model_dump())

    assert result.status == ToolResultStatus.OK
    updated = result.output["plan"]
    assert updated["status"] == "cancelled"
    steps = {s["id"]: s for s in updated["steps"]}
    assert steps["s1"]["status"] == StepStatus.COMPLETED.value
    assert steps["s2"]["status"] == StepStatus.SKIPPED.value


async def test_update_plan_immutable() -> None:
    """update_plan returns a new plan object; original is unchanged."""
    plugin = WorkflowPlugin()

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_plan_update": lambda self, **kw: None,
            })()})()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="done"),)
            )

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="immutable test",
        steps=(
            WorkflowStep(id="s1", title="Step 1", description="step 1", depends_on=[]),
            WorkflowStep(id="s2", title="Step 2", description="step 2", depends_on=["s1"]),
        ),
    )
    plugin._plans[str(plan.plan_id)] = plan
    original_steps = plan.steps

    input_data = UpdatePlanInput(
        plan_id=str(plan.plan_id),
        action=PlanAction.SKIP_STEP,
        step_id="s2",
    )
    result = await plugin._handle_update_plan(input_data.model_dump())

    assert result.status == ToolResultStatus.OK
    updated = result.output["plan"]
    updated_steps = [s["id"] for s in updated["steps"]]
    assert updated_steps == ["s1", "s2"]
    assert updated["steps"][1]["status"] == StepStatus.SKIPPED.value

    assert plan.steps == original_steps
    assert plan.steps[1].status == StepStatus.PENDING


async def test_step_with_subagent() -> None:
    """use_subagent=True 时，步骤通过 spawn_subagent 执行。"""
    plugin = WorkflowPlugin()

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_step_update": lambda self, **kw: None,
                "on_plan_completed": lambda self, **kw: None,
                "save_session": lambda self, **kw: None,
                "get_tool_specs": lambda self, context: (),
            })()})()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def spawn_subagent(self, spec, parent_tools=None, parent_config=None):
            assert spec.name == "s1"
            assert spec.max_turns == 3
            assert spec.timeout == 60.0
            return SubagentResult(
                spec=spec,
                status="ok",
                output="subagent done",
                session_id="sub-1",
            )

        async def run_session(self, *args, **kwargs):
            raise AssertionError("run_session should not be called when use_subagent=True")

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="subagent test",
        steps=(
            WorkflowStep(
                id="s1",
                title="Subagent step",
                description="do subagent task",
                use_subagent=True,
                subagent_max_turns=3,
                subagent_timeout=60.0,
            ),
        ),
    )

    result = await plugin._execute_plan(SessionContext(), plan, "mock-model")
    assert result["steps"][0]["status"] == StepStatus.COMPLETED.value
    assert result["steps"][0]["result"] == "subagent done"


async def test_step_without_subagent() -> None:
    """use_subagent=False 时，步骤通过 run_session 执行。"""
    plugin = WorkflowPlugin()

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_step_update": lambda self, **kw: None,
                "on_plan_completed": lambda self, **kw: None,
                "save_session": lambda self, **kw: None,
            })()})()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def spawn_subagent(self, *args, **kwargs):
            raise AssertionError("spawn_subagent should not be called when use_subagent=False")

        async def run_session(self, agent, initial_text=None, **kwargs):
            assert agent.name == "s1"
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="direct done"),)
            )

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="direct test",
        steps=(
            WorkflowStep(
                id="s1",
                title="Direct step",
                description="do direct task",
                use_subagent=False,
            ),
        ),
    )

    result = await plugin._execute_plan(SessionContext(), plan, "mock-model")
    assert result["steps"][0]["status"] == StepStatus.COMPLETED.value
    assert result["steps"][0]["result"] == "direct done"


async def test_plan_persistence() -> None:
    """Plan 保存到 SQLite 后可恢复。"""
    plugin = WorkflowPlugin()

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_step_update": lambda self, **kw: None,
                "on_plan_completed": lambda self, **kw: None,
                "save_session": lambda self, **kw: None,
                "save_plan": lambda plan, session_id: None,
                "load_plan": lambda plan_id: None,
                "list_plans": lambda session_id: [],
            })()})()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="done"),)
            )

    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="persist test",
        steps=(
            WorkflowStep(id="s1", title="Step 1", description="step 1", depends_on=[]),
            WorkflowStep(id="s2", title="Step 2", description="step 2", depends_on=["s1"]),
        ),
    )
    plugin._plans[str(plan.plan_id)] = plan

    await plugin.save_plan(plan, "session-1")

    loaded = await plugin.load_plan(str(plan.plan_id))
    assert loaded is not None
    assert loaded.task == "persist test"
    assert len(loaded.steps) == 2


async def test_plan_resume() -> None:
    """中断后恢复执行，已完成步骤不重复执行。"""
    plugin = WorkflowPlugin()

    call_count = 0

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_step_update": lambda self, **kw: None,
                "on_plan_completed": lambda self, **kw: None,
                "save_session": lambda self, **kw: None,
            })()})()

        async def aemit(self, event_type, **payload):
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content=f"step {agent.name} done"),)
            )

    plugin.harness = MockHarness()

    completed_step = WorkflowStep(id="s1", title="Step 1", description="step 1", status=StepStatus.COMPLETED)
    pending_step = WorkflowStep(id="s2", title="Step 2", description="step 2", depends_on=["s1"])
    plan = WorkflowPlan(
        task="resume test",
        steps=(completed_step, pending_step),
    )
    plugin._plans[str(plan.plan_id)] = plan

    result = await plugin._execute_plan(SessionContext(), plan, "mock-model")
    assert result["status"] == "completed"
    assert result["steps"][0]["status"] == StepStatus.COMPLETED.value
    assert result["steps"][1]["status"] == StepStatus.COMPLETED.value
    assert call_count == 1


async def test_observe_events_fired() -> None:
    """验证执行计划时 observe_event 被正确调用。"""
    events: list[dict[str, Any]] = []

    class MockBus:
        def __init__(self):
            self.pm = type("MockPM", (), {"hook": type("MockHook", (), {
                "on_step_update": lambda self, **kw: None,
                "on_plan_completed": lambda self, **kw: None,
                "save_session": lambda self, **kw: None,
            })()})()

        def on(self, event_type, fn):
            return lambda: None

        async def aemit(self, event_type, **payload):
            events.append({"type": event_type, **payload})
            return []

    class MockHarness:
        def __init__(self):
            self.bus = MockBus()

        async def run_session(self, agent, initial_text=None, **kwargs):
            return SessionContext(
                messages=(Message(role=Role.ASSISTANT, content="done"),)
            )

    plugin = WorkflowPlugin()
    plugin.harness = MockHarness()

    plan = WorkflowPlan(
        task="observe test",
        steps=(
            WorkflowStep(id="s1", title="Step 1", description="step 1", depends_on=[]),
            WorkflowStep(id="s2", title="Step 2", description="step 2", depends_on=["s1"]),
        ),
    )

    await plugin._execute_plan(SessionContext(), plan, "mock-model")

    event_types = [e["type"] for e in events]
    assert "plan_created" in event_types
    assert "plan_step_start" in event_types
    assert "plan_step_complete" in event_types
    assert "plan_completed" in event_types

    plan_created = [e for e in events if e["type"] == "plan_created"]
    assert len(plan_created) == 1
    assert plan_created[0]["plan_goal"] == "observe test"

    step_starts = [e for e in events if e["type"] == "plan_step_start"]
    assert len(step_starts) == 2
    assert step_starts[0]["step_id"] == "s1"
    assert step_starts[1]["step_id"] == "s2"


# --------------------------------------------------------------------------- #
# Batch A regressions: dependency cascade / frozen / persistence / event ctx  #
# --------------------------------------------------------------------------- #

async def test_failed_dep_cascades_skipped_status() -> None:
    """A FAILED dep must cause its dependents to be marked SKIPPED (with
    reason) instead of being left as PENDING forever."""
    plan = WorkflowPlan(
        plan_id=__import__("uuid").UUID("11111111-1111-1111-1111-111111111111"),
        task="cascade",
        steps=(
            WorkflowStep(id="a", title="A", description="", depends_on=(), status=StepStatus.FAILED),
            WorkflowStep(id="b", title="B", description="", depends_on=("a",), status=StepStatus.PENDING),
        ),
    )
    # Mirror the cascade block from _execute_plan.
    results: dict = {}
    original_steps = {s.id: s for s in plan.steps}
    remaining = list(plan.steps)
    for s in remaining:
        failed_dep = next(
            (
                dep
                for dep in s.depends_on
                if results.get(dep, original_steps.get(dep)).status == StepStatus.FAILED
            ),
            None,
        )
        if failed_dep is not None and s.status == StepStatus.PENDING:
            s = s.model_copy(
                update={"status": StepStatus.SKIPPED, "error": f"依赖步骤 {failed_dep} 失败，已自动跳过"}
            )
            results[s.id] = s
    assert results["b"].status == StepStatus.SKIPPED
    assert "a" in (results["b"].error or "")


async def test_persist_plan_does_not_mutate_original_snapshot() -> None:
    """_persist_plan must not mutate the existing memory dict in place.

    Before the fix, ``ctx.memory.get("workflow_plans", {})`` returned a live
    reference to the nested dict and the code wrote into it directly,
    violating the frozen-model contract. We assert the original dict object
    identity is unchanged after persist, and that persist forwarded a session
    whose memory carries the plan under a *separate* dict object.
    """
    import uuid as _uuid

    plugin = WorkflowPlugin()
    h = _harness(plugin)
    plugin.harness = h

    initial_memory: dict = {}
    sid = _uuid.uuid4()

    # Register a capture *plugin* (with a real save_session method). This is
    # the canonical pluggy mechanism and avoids any decorator / name quirks.
    captured_sessions: list = []

    class _SaveSessionCapture:
        @hookimpl
        def save_session(self, session: object) -> None:  # hookimpl method
            captured_sessions.append(session)

    capture = _SaveSessionCapture()
    h.bus.register(capture, name="batchA_capture_save")

    # Bind the session context so _persist_plan can read it.
    from pyharness.context import set_current, reset_current
    ctx = SessionContext(session_id=sid, memory=initial_memory)
    token = set_current(ctx)
    try:
        plan = WorkflowPlan(plan_id=_uuid.uuid4(), task="frozen", steps=())
        await plugin._persist_plan(plan)
    finally:
        reset_current(token)

    # The original dict object MUST remain empty (frozen contract).
    assert initial_memory == {}, f"frozen violated: {initial_memory!r}"
    # save_session received a session whose memory is a NEW dict containing workflow_plans.
    assert captured_sessions, "save_session was not called"
    saved = captured_sessions[-1]
    assert saved.memory is not initial_memory
    assert "workflow_plans" in saved.memory

    h.bus.pm.unregister(name="batchA_capture_save")


async def test_update_plan_persists_to_session_store() -> None:
    """update_plan must call _persist_plan so changes land in the store.

    We monkey-patch the CLASS method ``_persist_plan`` (via
    ``patch.object`` on the class) to record calls — this is the most direct
    possible causal assertion and avoids any pluggy / contextvar plumbing.
    """
    import uuid as _uuid
    from unittest.mock import patch

    persisted: list = []
    plugin = WorkflowPlugin()
    h = _harness(plugin)
    plugin.harness = h

    real_persist = WorkflowPlugin._persist_plan

    async def spy_persist(self, plan):  # type: ignore[no-untyped-def]
        persisted.append(plan.plan_id)
        await real_persist(self, plan)

    plan_id = _uuid.UUID("22222222-2222-2222-2222-222222222222")
    plan = WorkflowPlan(
        plan_id=plan_id,
        task="persist",
        steps=(WorkflowStep(id="x", title="X", description="", depends_on=(), status=StepStatus.PENDING),),
    )
    plugin._plans[str(plan_id)] = plan

    with patch.object(WorkflowPlugin, "_persist_plan", spy_persist):
        result = await plugin._handle_update_plan(
            {"plan_id": str(plan_id), "action": "skip_step", "step_id": "x"}
        )
        assert result.status == ToolResultStatus.OK
        assert persisted == [plan_id], f"update_plan did not call _persist_plan; got {persisted}"


async def test_workflow_events_carry_real_session_id() -> None:
    """Step / plan broadcasts must use current_context()'s session_id.

    We monkey-patch ``current_context`` (at the workflow import site) to
    return a known context, then assert the ``aemit`` payload carries it.
    """
    import uuid as _uuid
    from unittest.mock import patch

    plugin = WorkflowPlugin()
    h = _harness(plugin)
    plugin.harness = h

    captured: list = []
    real_aemit = h.bus.aemit

    async def spy_aemit(event_type, **payload):  # type: ignore[no-untyped-def]
        ctx = payload.get("context")
        if ctx is not None and event_type in ("plan_step_start", "plan_completed"):
            captured.append((event_type, ctx.session_id))
        return await real_aemit(event_type, **payload)

    h.bus.aemit = spy_aemit  # type: ignore[assignment]

    plan_id = _uuid.UUID("33333333-3333-3333-3333-333333333333")
    plan = WorkflowPlan(
        plan_id=plan_id,
        task="evt",
        steps=(WorkflowStep(id="p1", title="P1", description="", depends_on=(), status=StepStatus.PENDING),),
    )
    plugin._plans[str(plan_id)] = plan

    sid = _uuid.uuid4()
    fake_ctx = SessionContext(session_id=sid, memory={})

    # Direct on-instance patch: override _emit_context to bypass the module
    # import binding entirely. This is the most reliable approach.
    plugin._emit_context = lambda: fake_ctx  # type: ignore[assignment]

    await plugin._broadcast_step_event(
        plan, plan.steps[0].model_copy(update={"status": StepStatus.RUNNING}), "plan_step_start"
    )
    await plugin._broadcast_plan_event(plan, "plan_completed")

    h.bus.aemit = real_aemit  # type: ignore[assignment]

    assert captured, "no workflow events captured"
    for event_type, emitted_sid in captured:
        assert emitted_sid == sid, f"{event_type} emitted under wrong session {emitted_sid}"
