"""Tests for the python_exec tool plugin."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from pyharness.plugins.tool_python_exec import PythonExecPlugin, _check_docker_available
from pyharness.schema import ToolResultStatus


@pytest.fixture()
def plugin() -> PythonExecPlugin:
    return PythonExecPlugin()


@pytest.mark.asyncio
async def test_python_exec_success(plugin: PythonExecPlugin, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful execution returns stdout and OK status (explicit subprocess mode)."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "subprocess")
    result = await plugin.execute_tool(
        context=None,
        tool=type("ToolSpec", (), {"name": "python_exec"})(),
        arguments={"code": "print(1 + 1)"},
    )
    assert result is not None
    assert result.status == ToolResultStatus.OK
    assert "2" in result.output["stdout"]


@pytest.mark.asyncio
async def test_python_exec_timeout(plugin: PythonExecPlugin, monkeypatch: pytest.MonkeyPatch) -> None:
    """Infinite loop should timeout after 30s."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "subprocess")
    import asyncio

    result = await asyncio.wait_for(
        plugin.execute_tool(
            context=None,
            tool=type("ToolSpec", (), {"name": "python_exec"})(),
            arguments={"code": "while True: pass"},
        ),
        timeout=35,
    )
    assert result is not None
    assert result.status == ToolResultStatus.ERROR
    assert "超时" in result.error


@pytest.mark.asyncio
async def test_python_exec_syntax_error(plugin: PythonExecPlugin, monkeypatch: pytest.MonkeyPatch) -> None:
    """Syntax errors should return ERROR with stderr."""
    monkeypatch.setenv("PYHARNESS_SANDBOX", "subprocess")
    result = await plugin.execute_tool(
        context=None,
        tool=type("ToolSpec", (), {"name": "python_exec"})(),
        arguments={"code": "print("},
    )
    assert result is not None
    assert result.status == ToolResultStatus.ERROR
    assert result.output.get("returncode") != 0


@pytest.mark.asyncio
async def test_python_exec_missing_code(plugin: PythonExecPlugin) -> None:
    """Missing code parameter returns ERROR."""
    result = await plugin.execute_tool(
        context=None,
        tool=type("ToolSpec", (), {"name": "python_exec"})(),
        arguments={},
    )
    assert result is not None
    assert result.status == ToolResultStatus.ERROR
    assert "缺少 'code' 参数" in result.error


@pytest.mark.asyncio
async def test_python_exec_empty_code(plugin: PythonExecPlugin) -> None:
    """Empty code string returns ERROR."""
    result = await plugin.execute_tool(
        context=None,
        tool=type("ToolSpec", (), {"name": "python_exec"})(),
        arguments={"code": "   "},
    )
    assert result is not None
    assert result.status == ToolResultStatus.ERROR
    assert "缺少 'code' 参数" in result.error


def pytest_configure(config):
    """Register docker marker."""
    config.addinivalue_line("markers", "docker: mark test as requiring Docker")


@pytest.mark.docker
@pytest.mark.asyncio
async def test_docker_mode_echo_success() -> None:
    """Docker mode: simple echo should succeed."""
    if not await _check_docker_available():
        pytest.skip("Docker is not available")

    plugin = PythonExecPlugin()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYHARNESS_SANDBOX", "docker")
        result = await plugin.execute_tool(
            context=None,
            tool=type("ToolSpec", (), {"name": "python_exec"})(),
            arguments={"code": "print('hello from docker')"},
        )
    assert result is not None
    assert result.status == ToolResultStatus.OK
    assert "hello from docker" in result.output["stdout"]
    assert result.output.get("backend") == "docker"


@pytest.mark.docker
@pytest.mark.asyncio
async def test_docker_mode_timeout_kills_container() -> None:
    """Docker mode: infinite loop should timeout and kill container."""
    if not await _check_docker_available():
        pytest.skip("Docker is not available")

    plugin = PythonExecPlugin()
    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYHARNESS_SANDBOX", "docker")
        result = await asyncio.wait_for(
            plugin.execute_tool(
                context=None,
                tool=type("ToolSpec", (), {"name": "python_exec"})(),
                arguments={"code": "while True: pass"},
            ),
            timeout=35,
        )
    assert result is not None
    assert result.status == ToolResultStatus.ERROR
    assert "超时" in result.error


@pytest.mark.docker
@pytest.mark.asyncio
async def test_docker_mode_network_disabled() -> None:
    """Docker mode: network access should be disabled."""
    if not await _check_docker_available():
        pytest.skip("Docker is not available")

    plugin = PythonExecPlugin()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYHARNESS_SANDBOX", "docker")
        result = await plugin.execute_tool(
            context=None,
            tool=type("ToolSpec", (), {"name": "python_exec"})(),
            arguments={"code": "import socket; print(socket.gethostbyname('google.com'))"},
        )
    assert result is not None
    assert result.status == ToolResultStatus.ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
