"""Security regression tests for the python_exec tool.

These pin down the sandbox contract introduced after the code review:

* ``auto`` mode is **fail-closed** — with Docker unavailable it must refuse to
  execute rather than silently falling back to an unsandboxed host process.
* Oversized code payloads are rejected before any execution.
* Docker argv is built via named containers (so a timeout can ``docker kill``
  the *container*, not the docker *client* pid).
* Subprocess-mode results are always marked ``sandbox: False`` so the guard /
  approval layer can force human confirmation.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import pytest

import pyharness.plugins.tool_python_exec as tpe
from pyharness.plugins.tool_python_exec import (
    _MAX_CODE_LENGTH,
    PythonExecPlugin,
    _docker_kill_cmd,
    _docker_run_cmd,
)
from pyharness.schema import ToolResultStatus


@pytest.fixture()
def plugin() -> PythonExecPlugin:
    return PythonExecPlugin()


def _spec() -> Any:
    return type("ToolSpec", (), {"name": "python_exec"})()


# --------------------------------------------------------------------------- #
# fail-closed auto mode
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_auto_mode_fails_closed_without_docker(plugin: PythonExecPlugin, monkeypatch: pytest.MonkeyPatch) -> None:
    """Docker unavailable + auto mode => refuse to execute (no silent fallback)."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "auto")
    async def _no_docker() -> bool:
        return False
    monkeypatch.setattr(plugin, "_ensure_docker", _no_docker)

    result = await plugin.execute_tool(context=None, tool=_spec(), arguments={"code": "print(1)"})

    assert result is not None
    assert result.status == ToolResultStatus.ERROR
    assert "无可用沙箱" in (result.error or "")
    assert "PYHARNESS_SANDBOX=subprocess" in (result.error or "")
    assert result.output.get("backend") == "none"
    assert result.output.get("sandbox") is False


@pytest.mark.asyncio
async def test_explicit_subprocess_mode_is_high_risk_opt_in(
    plugin: PythonExecPlugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit opt-in still works, but the result must be flagged unsandboxed."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "subprocess")
    result = await plugin.execute_tool(
        context=None, tool=_spec(), arguments={"code": "print('ok')"}
    )
    assert result is not None
    assert result.status == ToolResultStatus.OK
    assert result.output.get("sandbox") is False
    assert result.output.get("backend") == "subprocess"


# --------------------------------------------------------------------------- #
# payload limits
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_oversized_code_is_rejected(plugin: PythonExecPlugin, monkeypatch: pytest.MonkeyPatch) -> None:
    """Code above the size cap is rejected before any execution backend runs."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "subprocess")
    called = False

    async def _spy(code: str) -> Any:  # noqa: ARG001
        nonlocal called
        called = True
        raise AssertionError("must not reach the execution backend")

    monkeypatch.setattr(tpe, "_execute_in_subprocess", _spy)

    result = await plugin.execute_tool(
        context=None, tool=_spec(), arguments={"code": "x = 1" * (_MAX_CODE_LENGTH // 4 + 1)}
    )

    assert result is not None
    assert result.status == ToolResultStatus.ERROR
    assert "代码过长" in (result.error or "")
    assert called is False


# --------------------------------------------------------------------------- #
# named docker containers
# --------------------------------------------------------------------------- #
def test_docker_run_cmd_uses_named_container(tmp_path: Path) -> None:
    """The docker run argv carries a unique container name (killable later)."""
    name = "pyharness_exec_abc123"
    cmd = _docker_run_cmd(name, tmp_path)
    assert cmd[0:2] == ["docker", "run"]
    assert "--name" in cmd
    assert cmd[cmd.index("--name") + 1] == name
    # hardening flags stay intact
    for flag in ("--network", "--memory", "--cpus", "--read-only", "--pids-limit"):
        assert flag in cmd


def test_docker_kill_cmd_targets_container_name() -> None:
    """docker kill must target the container name, not the client pid."""
    assert _docker_kill_cmd("pyharness_exec_abc123") == ["docker", "kill", "pyharness_exec_abc123"]


@pytest.mark.docker
@pytest.mark.asyncio
async def test_docker_timeout_uses_named_kill(plugin: PythonExecPlugin, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a hung container is killed by name (requires Docker)."""
    if os.getenv("PYHARNESS_DOCKER_TESTS", "") != "1":
        pytest.skip("set PYHARNESS_DOCKER_TESTS=1 to run docker-kill integration test")
    if not await tpe._check_docker_available():
        pytest.skip("Docker is not available")

    monkeypatch.setenv("PYHARNESS_SANDBOX", "docker")
    result = await plugin.execute_tool(
        context=None, tool=_spec(), arguments={"code": "while True: pass"}
    )
    assert result is not None
    assert result.status == ToolResultStatus.ERROR
    assert "超时" in (result.error or "")
    assert result.output.get("sandbox") is True


def test_module_reloads_cleanly() -> None:
    """Sanity: the refactored module still exposes its public surface."""
    mod = importlib.reload(tpe)
    assert hasattr(mod, "PythonExecPlugin")
    assert hasattr(mod, "_docker_run_cmd")
    assert hasattr(mod, "_MAX_CODE_LENGTH")