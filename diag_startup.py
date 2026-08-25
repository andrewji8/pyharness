"""诊断 PyHarness 启动卡点：逐模块 import + Harness 构建计时（带 flush）。"""
from __future__ import annotations

import sys
import time


def t0() -> float:
    return time.monotonic()


def log(name: str, start: float) -> None:
    print(f"[+{round(time.monotonic() - start, 2):>6}s] {name}", flush=True)


def main() -> None:
    start = t0()
    log("脚本启动", start)

    modules = [
        "pyharness",
        "pyharness.context",
        "pyharness.schema",
        "pyharness.specs",
        "pyharness.llm",
        "pyharness.plugins.mcp_client",
        "pyharness.plugins.mcp_config",
        "pyharness.plugins.tool_web",
        "pyharness.plugins.tool_fs",
        "pyharness.plugins.workflow",
        "pyharness.plugins.tool_subagent",
        "pyharness.plugins.session_store",
        "pyharness.plugins.guard_approval",
        "pyharness.plugins.context_compaction",
        "pyharness.plugins.embedding",
        "pyharness.plugins.vector_store",
        "pyharness.plugins.tool_knowledge",
        "pyharness.plugins.ui_cli",
        "pyharness.plugins.ui_websocket",
        "pyharness.plugins.web_ui",
    ]
    for name in modules:
        m0 = time.monotonic()
        try:
            __import__(name)
            log(f"import ok  {name}", m0)
        except Exception as exc:  # noqa: BLE001
            log(f"IMPORT ERR {name}: {type(exc).__name__}: {exc}", m0)

    log("--- 构建 Harness() ---", start)
    from pyharness.core import Harness

    h0 = time.monotonic()
    harness = Harness()
    log(f"Harness() 构建完成 {len(harness.bus.pm.get_plugins())} 插件", h0)

    log("--- 显式注册 extra 插件 ---", start)
    from pyharness.plugins.cli.app import _harness

    x0 = time.monotonic()
    h = _harness(verbose=False)
    log("_harness() 完成", x0)

    log("--- initialize() ---", start)
    i0 = time.monotonic()
    h.initialize()
    log("initialize() 完成", i0)
    log("全部完成", start)


if __name__ == "__main__":
    main()