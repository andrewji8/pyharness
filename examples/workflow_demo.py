"""端到端 Workflow 集成示例。

场景：Agent 使用 Plan 完成一个多步骤任务：
"调研 Python 3.12 和 Rust 异步运行时，然后写一份对比报告"。

演示流程：
1. 初始化 Harness 并注册插件
2. 通过 workflow_execute 生成并执行计划
3. 通过 update_plan 动态修改计划
4. 通过 get_plan_status 查看状态
"""

from __future__ import annotations

import asyncio

from pyharness import Harness
from pyharness.context import SessionContext
from pyharness.core import _settle
from pyharness.plugins.llm import entry as llm
from pyharness.plugins.workflow import WorkflowPlugin
from pyharness.schema import AgentConfig, HarnessConfig, LLMResponse, PlanAction, StepStatus, ToolResultStatus, UpdatePlanInput, WorkflowPlan, WorkflowStep


async def main() -> None:
    # 1. 初始化 Harness
    h = Harness(config=HarnessConfig(auto_load_entry_points=False))
    llm.clear()
    llm.use_dummy(models=("mock-model",), plan=[LLMResponse(model="mock-model", content="done")])
    h.register_plugin(llm)
    h.register_plugin(WorkflowPlugin())
    h.initialize()

    ctx = SessionContext()

    specs = {}
    for plugin_specs in h.bus.pm.hook.get_tool_specs(context=ctx):
        specs.update({s.name: s for s in plugin_specs})

    # 2. 生成并执行计划
    print(f"\n{'='*60}")
    print("步骤 1: 生成并执行计划")
    print(f"{'='*60}")

    llm.clear()
    llm.use_dummy(
        models=("mock-model",),
        plan=[
            LLMResponse(
                model="mock-model",
                content='''[
                    {"id": "s1", "title": "调研 Python 3.12", "description": "Research Python 3.12 async features", "depends_on": []},
                    {"id": "s2", "title": "调研 Tokio", "description": "Research Rust Tokio runtime", "depends_on": []},
                    {"id": "s3", "title": "调研 async-std", "description": "Research Rust async-std runtime", "depends_on": []},
                    {"id": "s4", "title": "写对比报告", "description": "Write a comparison report", "depends_on": ["s1", "s2", "s3"]}
                ]''',
            ),
            LLMResponse(model="mock-model", content="Python 3.12 async features researched"),
            LLMResponse(model="mock-model", content="Rust Tokio runtime researched"),
            LLMResponse(model="mock-model", content="Rust async-std runtime researched"),
            LLMResponse(model="mock-model", content="Comparison report written"),
        ],
    )

    execute_tool = specs["workflow_execute"]
    raw = h.bus.pm.hook.execute_tool(
        context=ctx,
        tool=execute_tool,
        arguments={"task": "调研 Python 3.12 和 Rust 异步运行时，写对比报告", "model": "mock-model"},
    )
    result = next((r for r in await _settle(raw) if r is not None), None)

    if result is None or result.status != ToolResultStatus.OK:
        print(f"工作流执行失败: {result.error if result else 'No result'}")
        return

    plan_id = result.output["plan_id"]
    print(f"\n计划执行完成!")
    print(f"Plan ID: {plan_id}")
    print(f"状态: {result.output['status']}")
    print(f"\n步骤执行结果:")
    for step in result.output["steps"]:
        deps = ", ".join(step["depends_on"]) if step["depends_on"] else "无"
        print(f"  [{step['id']}] {step['title']} - 依赖: [{deps}] -> {step['status']}")
        if step.get("result"):
            print(f"    结果: {step['result']}")

    # 3. 动态修改：插入新步骤（在已完成的计划上演示 update_plan 功能）
    print(f"\n{'='*60}")
    print("步骤 2: 动态修改计划 - 演示 update_plan 工具")
    print(f"{'='*60}")

    update_tool = specs["update_plan"]
    update_input = UpdatePlanInput(
        plan_id=plan_id,
        action=PlanAction.ADD_STEP,
        step_id="s3b",
        new_title="调研 Trio",
        new_description="Research Python Trio async runtime",
        insert_after="s1",
        reason="演示动态添加步骤",
    )
    raw_update = h.bus.pm.hook.execute_tool(
        context=ctx, tool=update_tool, arguments=update_input.model_dump()
    )
    update_result = next((r for r in await _settle(raw_update) if r is not None), None)

    if update_result and update_result.status == ToolResultStatus.OK:
        updated_plan = update_result.output["plan"]
        print(f"\n修改成功! 注意: 由于原计划已完成，新增步骤处于 PENDING 状态")
        print(f"新步骤列表:")
        for step in updated_plan["steps"]:
            deps = ", ".join(step["depends_on"]) if step["depends_on"] else "无"
            print(f"  [{step['id']}] {step['title']} - 依赖: [{deps}] - 状态: {step['status']}")
    else:
        print(f"修改结果: {update_result.error if update_result else 'No result'}")
        print("(这是预期的，因为计划已完成，无法修改已完成/取消的计划)")

    # 4. 跳过步骤演示
    print(f"\n{'='*60}")
    print("步骤 3: 跳过步骤演示")
    print(f"{'='*60}")

    # 创建一个未完成的计划来演示 skip
    pending_plan = WorkflowPlan(
        task="跳过步骤演示",
        steps=(
            WorkflowStep(id="s1", title="步骤 1", description="step 1", depends_on=[]),
            WorkflowStep(id="s2", title="步骤 2", description="step 2", depends_on=[]),
            WorkflowStep(id="s3", title="步骤 3", description="step 3", depends_on=["s2"]),
        ),
    )

    # 先执行第一步
    llm.clear()
    llm.use_dummy(
        models=("mock-model",),
        plan=[
            LLMResponse(
                model="mock-model",
                content='[{"id": "s1", "title": "步骤 1", "description": "step 1", "depends_on": []}, '
                        '{"id": "s2", "title": "步骤 2", "description": "step 2", "depends_on": []}, '
                        '{"id": "s3", "title": "步骤 3", "description": "step 3", "depends_on": ["s2"]}]',
            ),
            LLMResponse(model="mock-model", content="step 1 done"),
        ],
    )
    raw2 = h.bus.pm.hook.execute_tool(
        context=ctx, tool=execute_tool, arguments={"task": pending_plan.task, "model": "mock-model"}
    )
    result2 = next((r for r in await _settle(raw2) if r is not None), None)

    if result2 and result2.status == ToolResultStatus.OK:
        print(f"部分执行完成，Plan ID: {result2.output['plan_id']}")
        partial_plan_id = result2.output["plan_id"]

        # 跳过 s2
        skip_input = UpdatePlanInput(
            plan_id=partial_plan_id,
            action=PlanAction.SKIP_STEP,
            step_id="s2",
            reason="用户决定跳过",
        )
        raw_skip = h.bus.pm.hook.execute_tool(
            context=ctx, tool=update_tool, arguments=skip_input.model_dump()
        )
        skip_result = next((r for r in await _settle(raw_skip) if r is not None), None)

        if skip_result and skip_result.status == ToolResultStatus.OK:
            skipped_plan = skip_result.output["plan"]
            steps = {s["id"]: s for s in skipped_plan["steps"]}
            print(f"\n跳过 s2 后的计划状态:")
            print(f"  s1: {steps['s1']['status']}")
            print(f"  s2: {steps['s2']['status']}")
            print(f"  s3: {steps['s3']['status']}")
        else:
            print(f"跳过失败: {skip_result.error if skip_result else 'No result'}")

    # 5. 查看最终状态
    print(f"\n{'='*60}")
    print("步骤 4: 查看最终状态")
    print(f"{'='*60}")

    status_tool = specs["get_plan_status"]
    raw_status = h.bus.pm.hook.execute_tool(
        context=ctx, tool=status_tool, arguments={"plan_id": plan_id}
    )
    status_result = next((r for r in await _settle(raw_status) if r is not None), None)

    if status_result and status_result.status == ToolResultStatus.OK:
        output = status_result.output
        print(f"\nPlan ID: {output['plan_id']}")
        print(f"任务: {output['task']}")
        print(f"状态: {output['status']}")
        print(f"步骤总数: {len(output['steps'])}")
        completed = sum(1 for s in output["steps"] if s["status"] == StepStatus.COMPLETED.value)
        print(f"已完成: {completed}/{len(output['steps'])}")
        print("\n步骤详情:")
        for step in output["steps"]:
            print(f"  [{step['id']}] {step['title']} -> {step['status']}")
    else:
        print(f"查询失败: {status_result.error if status_result else 'No result'}")

    print(f"\n{'='*60}")
    print("演示完成！")
    print("关键特性:")
    print("  1. 计划生成: LLM 生成结构化 JSON 计划")
    print("  2. 并行执行: 无依赖步骤并行执行")
    print("  3. 依赖管理: 依赖步骤串行等待")
    print("  4. 动态修改: update_plan 支持 ADD_STEP/SKIP_STEP 等操作")
    print("  5. 状态查询: get_plan_status 实时查看进度")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
