"""诊断2：实际执行一次 run_session，定位卡在哪个 hook（写文件日志）。"""
from __future__ import annotations

import asyncio
import time

_LOG = open("diag_session.log", "w", encoding="utf-8")


def emit(name: str, start: float) -> None:
    line = f"[+{round(time.monotonic() - start, 2):>6}s] {name}"
    print(line, flush=True)
    _LOG.write(line + "\n")
    _LOG.flush()


async def main() -> None:
    start = time.monotonic()
    from pyharness.plugins.llm import entry as llm
    from pyharness.schema import LLMResponse

    emit("配置 dummy provider", start)
    llm.clear()
    llm.use_dummy(models=("demo",), plan=[LLMResponse(model="demo", content="OK")])

    emit("构建 _harness()", start)
    from pyharness.plugins.cli.app import _harness

    h = _harness(verbose=False)
    emit("harness 就绪，枚举 get_tool_specs", start)

    from pyharness.schema import AgentConfig
    from pyharness.context import open_session

    async with open_session() as ctx:
        g0 = time.monotonic()
        try:
            groups = h.bus.pm.hook.get_tool_specs(context=ctx)
            names = []
            for grp in groups or []:
                for s in grp or []:
                    names.append(getattr(s, "name", "?"))
            emit(f"  get_tool_specs -> {len(names)} 工具: {sorted(set(names))}", g0)
        except Exception as exc:  # noqa: BLE001
            emit(f"  get_tool_specs ERR: {type(exc).__name__}: {exc}", g0)

        emit("开始 run_session", start)
        r0 = time.monotonic()
        try:
            final = await h.run_session(AgentConfig(name="a", model="demo"), initial_text="hi")
            emit(f"  run_session -> {len(final.messages)} msgs in {round(time.monotonic()-r0,2)}s", r0)
        except Exception as exc:  # noqa: BLE001
            emit(f"  run_session ERR: {type(exc).__name__}: {exc}", r0)

    emit("完成", start)
    _LOG.close()


asyncio.run(main())