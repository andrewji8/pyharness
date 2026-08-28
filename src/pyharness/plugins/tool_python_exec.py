"""Python code execution tool plugin for PyHarness.

Provides the ``python_exec`` tool, allowing agents to run Python code in an
isolated subprocess or Docker sandbox and capture stdout/stderr.

Security
--------
* Subprocess mode: code runs in a short-lived ``python -c`` subprocess.
* Docker mode: code runs in a ``python:3.11-slim`` container with:
  --rm --network none --memory 256m --memory-swap 256m
  --cpus 0.5 --pids-limit 64 --read-only --tmpfs /tmp
* A hard timeout prevents runaway execution in both modes.
* Only the captured output is returned to the agent; no host state is exposed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import ToolArg, ToolResult, ToolResultStatus, ToolSpec

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")

# Hard timeout for code execution.
_EXEC_TIMEOUT = 30.0

# Upper bound on submitted code size (guards against oversized payloads).
_MAX_CODE_LENGTH = 100_000

# Docker image for sandbox execution.
_DOCKER_IMAGE = "python:3.11-slim"
_DOCKER_TIMEOUT = "30s"

# Workspace root for Docker mode.
_WORKSPACE_ROOT = Path(".workspace")


def _get_sandbox_mode() -> str:
    """Determine execution backend from PYHARNESS_SANDBOX env var."""
    mode = os.getenv("PYHARNESS_SANDBOX", "auto").lower()
    if mode in ("docker", "subprocess"):
        return mode
    if mode == "auto":
        return "auto"
    logger.warning("Unknown PYHARNESS_SANDBOX=%r, falling back to auto", mode)
    return "auto"


async def _check_docker_available() -> bool:
    """Check if Docker is available by running 'docker info'."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
        return proc.returncode == 0
    except (FileNotFoundError, asyncio.TimeoutError, Exception):
        return False


async def _ensure_docker_image() -> bool:
    """Pull the Docker image if not present. Returns True on success."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "image", "inspect", _DOCKER_IMAGE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=10.0)
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
        if proc.returncode == 0:
            return True
    except Exception:
        pass

    logger.info("Pulling Docker image %s...", _DOCKER_IMAGE)
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "pull", _DOCKER_IMAGE,
            stdout=asyncio.subprocess.STDOUT,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
        if proc.returncode == 0:
            logger.info("Docker image %s pulled successfully", _DOCKER_IMAGE)
            return True
        logger.warning("Docker pull failed: %s", stdout.decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.warning("Docker pull error: %s", exc)
    return False


def _docker_run_cmd(container_name: str, workdir: Path) -> list[str]:
    """Build the ``docker run`` argv for a named, locked-down container."""
    return [
        "docker", "run", "--rm",
        "--name", container_name,
        "--network", "none",
        "--memory", "256m", "--memory-swap", "256m",
        "--cpus", "0.5", "--pids-limit", "64",
        "--read-only",
        "--tmpfs", "/tmp",
        "-v", f"{workdir}:/work",
        "-w", "/work",
        _DOCKER_IMAGE,
        "python", "/work/__task__.py",
    ]


def _docker_kill_cmd(container_name: str) -> list[str]:
    """Build the ``docker kill`` argv for a named container."""
    return ["docker", "kill", container_name]


async def _execute_in_docker(code: str, workdir: Path) -> ToolResult:
    """Execute Python code in a Docker sandbox."""
    container_name = f"pyharness_exec_{uuid.uuid4().hex[:12]}"
    task_file = workdir / "__task__.py"
    try:
        task_file.write_text(code, encoding="utf-8")
    except Exception as exc:
        return ToolResult(
            tool_name="python_exec",
            status=ToolResultStatus.ERROR,
            error=f"Failed to write task file: {exc}",
            output={"code": code},
        )

    cmd = _docker_run_cmd(container_name, workdir)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_EXEC_TIMEOUT
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            # Kill by *container name*: proc.pid is the docker *client* pid,
            # not the container — killing it left the container running.
            try:
                await asyncio.create_subprocess_exec(
                    *_docker_kill_cmd(container_name),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except Exception:
                pass
            return ToolResult(
                tool_name="python_exec",
                status=ToolResultStatus.ERROR,
                error=f"执行超时（{_EXEC_TIMEOUT}s）",
                output={"code": code, "backend": "docker", "sandbox": True},
            )

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return ToolResult(
                tool_name="python_exec",
                status=ToolResultStatus.ERROR,
                error=stderr_text or f"进程退出码: {proc.returncode}",
                output={
                    "code": code,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "returncode": proc.returncode,
                    "backend": "docker",
                },
            )

        return ToolResult(
            tool_name="python_exec",
            status=ToolResultStatus.OK,
            output={
                "code": code,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "returncode": 0,
                "backend": "docker",
            },
        )
    except FileNotFoundError:
        return ToolResult(
            tool_name="python_exec",
            status=ToolResultStatus.ERROR,
            error="未找到 docker 可执行文件，请确保 Docker 已安装并在 PATH 中。",
            output={"code": code, "backend": "docker"},
        )
    except Exception as exc:
        logger.exception("docker execution failed")
        return ToolResult(
            tool_name="python_exec",
            status=ToolResultStatus.ERROR,
            error=f"Docker 执行失败: {type(exc).__name__}: {exc}",
            output={"code": code, "backend": "docker"},
        )
    finally:
        try:
            task_file.unlink(missing_ok=True)
        except Exception:
            pass


async def _execute_in_subprocess(code: str) -> ToolResult:
    """Execute Python code in a local subprocess."""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",  # isolated mode: ignore env/user-site side channels
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_EXEC_TIMEOUT
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ToolResult(
                tool_name="python_exec",
                status=ToolResultStatus.ERROR,
                error=f"执行超时（{_EXEC_TIMEOUT}s）",
                output={"code": code, "backend": "subprocess", "sandbox": False},
            )

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return ToolResult(
                tool_name="python_exec",
                status=ToolResultStatus.ERROR,
                error=stderr_text or f"进程退出码: {proc.returncode}",
                output={
                    "code": code,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "returncode": proc.returncode,
                    "backend": "subprocess",
                    "sandbox": False,
                },
            )

        return ToolResult(
            tool_name="python_exec",
            status=ToolResultStatus.OK,
            output={
                "code": code,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "returncode": 0,
                "backend": "subprocess",
                "sandbox": False,
            },
        )
    except FileNotFoundError:
        return ToolResult(
            tool_name="python_exec",
            status=ToolResultStatus.ERROR,
            error="无法启动 Python 解释器（sys.executable）。",
            output={"code": code, "backend": "subprocess", "sandbox": False},
        )
    except Exception as exc:
        logger.exception("subprocess execution failed")
        return ToolResult(
            tool_name="python_exec",
            status=ToolResultStatus.ERROR,
            error=f"执行失败: {type(exc).__name__}: {exc}",
            output={"code": code, "backend": "subprocess", "sandbox": False},
        )


def _python_exec_spec() -> ToolSpec:
    return ToolSpec(
        name="python_exec",
        description=(
            "Execute Python code in an isolated subprocess or Docker sandbox and return stdout/stderr. "
            "Use this for any computation, file parsing, math, or data transformation."
        ),
        parameters=(
            ToolArg(
                name="code",
                type="string",
                description="Python code to execute",
                required=True,
            ),
        ),
        timeout_seconds=_EXEC_TIMEOUT,
    )


class PythonExecPlugin:
    """Python code execution tool provider."""

    def __init__(self) -> None:
        self._docker_available: bool | None = None
        self._docker_image_ready: bool = False

    @hookimpl
    def get_tool_specs(self, context: SessionContext) -> tuple[ToolSpec, ...]:
        return (_python_exec_spec(),)

    @hookimpl
    async def execute_tool(
        self,
        context: SessionContext,
        tool: ToolSpec,
        arguments: dict[str, object],
    ) -> ToolResult | None:
        """Execute Python code in a subprocess or Docker sandbox."""
        if tool.name != "python_exec":
            return None

        code = str(arguments.get("code", "")).strip()
        if not code:
            return ToolResult(
                tool_name="python_exec",
                status=ToolResultStatus.ERROR,
                error="缺少 'code' 参数。",
                output={},
            )
        if len(code) > _MAX_CODE_LENGTH:
            return ToolResult(
                tool_name="python_exec",
                status=ToolResultStatus.ERROR,
                error=f"代码过长（{len(code)} 字符 > 上限 {_MAX_CODE_LENGTH}）。",
                output={"code_chars": len(code)},
            )

        mode = _get_sandbox_mode()
        if mode == "subprocess":
            return await _execute_in_subprocess(code)

        if mode == "docker":
            if not await self._ensure_docker():
                return ToolResult(
                    tool_name="python_exec",
                    status=ToolResultStatus.ERROR,
                    error="Docker 不可用，请安装 Docker 或设置 PYHARNESS_SANDBOX=subprocess",
                    output={"code": code},
                )
            workdir = _WORKSPACE_ROOT / "python_exec"
            workdir.mkdir(parents=True, exist_ok=True)
            return await _execute_in_docker(code, workdir)

        # auto mode: fail-closed. Never silently fall back to an unsandboxed
        # host process — that would defeat the sandbox contract entirely.
        if await self._ensure_docker():
            workdir = _WORKSPACE_ROOT / "python_exec"
            workdir.mkdir(parents=True, exist_ok=True)
            return await _execute_in_docker(code, workdir)
        return ToolResult(
            tool_name="python_exec",
            status=ToolResultStatus.ERROR,
            error=(
                "无可用沙箱：Docker 不可用，已拒绝执行（fail-closed）。"
                "如确需在宿主机直接执行（高危、无隔离），"
                "请显式设置环境变量 PYHARNESS_SANDBOX=subprocess。"
            ),
            output={"code_chars": len(code), "backend": "none", "sandbox": False},
        )

    async def _ensure_docker(self) -> bool:
        """Lazy-check Docker availability and image presence."""
        if self._docker_available is None:
            self._docker_available = await _check_docker_available()
            if not self._docker_available:
                logger.warning("Docker is not available, falling back to subprocess mode")
        if self._docker_available and not self._docker_image_ready:
            self._docker_image_ready = await _ensure_docker_image()
        return bool(self._docker_available and self._docker_image_ready)

    async def init_sandbox(self) -> bool:
        """Public API: initialize the Docker sandbox (pull image)."""
        if not await _check_docker_available():
            logger.warning("Docker is not available")
            return False
        return await _ensure_docker_image()


__all__ = ["PythonExecPlugin"]
