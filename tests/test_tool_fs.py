"""Regression tests for the sandboxed filesystem tool (tool_fs).

Pins down the hardening from the code review:

* Path traversal and absolute paths stay blocked.
* The workspace boundary check is case-insensitive (Windows-safe).
* ``fs_write`` is atomic (temp file + ``os.replace``, no ``.tmp`` residue) and
  enforces a per-call byte cap.
"""

from __future__ import annotations

import os

import pytest

from pyharness.plugins.tool_fs import FileSystemPlugin
from pyharness.schema import ToolResultStatus


@pytest.fixture()
def fs(tmp_path) -> FileSystemPlugin:
    return FileSystemPlugin(workspace_dir=str(tmp_path / "ws"), max_file_size=1024 * 1024)


def _sync(coro):
    import asyncio

    return asyncio.run(coro)


def test_traversal_blocked(fs: FileSystemPlugin) -> None:
    with pytest.raises(PermissionError):
        fs._resolve("../outside.txt")
    with pytest.raises(PermissionError):
        fs._resolve("a/../../outside.txt")


def test_absolute_path_blocked(fs: FileSystemPlugin) -> None:
    with pytest.raises(PermissionError):
        fs._resolve(os.path.abspath("evil.txt"))


def test_workspace_case_difference_stays_contained(tmp_path) -> None:
    """A workspace path built with different casing must not weaken the boundary."""
    ws = tmp_path / "WS"  # different case than what the plugin may compute
    plugin = FileSystemPlugin(workspace_dir=str(ws))
    resolved = plugin._resolve("sub/dir/file.txt")
    assert os.path.normcase(resolved).startswith(os.path.normcase(str(ws)))


@pytest.mark.asyncio
async def test_write_is_atomic_and_complete(tmp_path) -> None:
    plugin = FileSystemPlugin(workspace_dir=str(tmp_path / "ws"), max_write_bytes=1024)
    result = await plugin._write({"path": "notes/hello.txt", "content": "你好 PyHarness"})

    assert result.status == ToolResultStatus.OK
    target = tmp_path / "ws" / "notes" / "hello.txt"
    assert target.read_text(encoding="utf-8") == "你好 PyHarness"
    # no temp residue next to the target
    assert list(target.parent.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_write_over_cap_rejected(tmp_path) -> None:
    plugin = FileSystemPlugin(workspace_dir=str(tmp_path / "ws"), max_write_bytes=16)
    result = await plugin._write({"path": "big.txt", "content": "x" * 1024})

    assert result.status == ToolResultStatus.ERROR
    assert "写入内容过大" in (result.error or "")
    assert not (tmp_path / "ws" / "big.txt").exists()


@pytest.mark.asyncio
async def test_write_cap_disabled_with_zero(tmp_path) -> None:
    plugin = FileSystemPlugin(workspace_dir=str(tmp_path / "ws"), max_write_bytes=0)
    result = await plugin._write({"path": "big.txt", "content": "x" * 1024})
    assert result.status == ToolResultStatus.OK