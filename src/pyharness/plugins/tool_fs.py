"""File System tools plugin for PyHarness.

Provides safe, sandboxed file system operations for agents. All paths are
resolved against a configurable ``workspace_dir`` and validated to prevent
directory traversal attacks.

Security
--------
* Every user-supplied path is joined with ``workspace_dir`` and resolved via
  ``os.path.realpath()``.
* The resolved path MUST start with ``workspace_dir``; otherwise a
  ``PermissionError`` is raised.
* Symbolic links are followed by ``realpath`` and re-checked against the
  workspace boundary.
* Absolute paths and empty inputs are rejected before joining.
* File reads are capped at ``MAX_FILE_SIZE`` (default 1 MB) to prevent
  token-context explosions from huge files.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import ToolArg, ToolResult, ToolResultStatus

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")

# Default workspace: a ``.workspace`` folder next to the current working dir.
_DEFAULT_WORKSPACE = os.path.abspath(".workspace")

# Hard cap on read size: 1 MB.
MAX_FILE_SIZE: int = 1 * 1024 * 1024


class FileSystemPlugin:
    """Sandboxed file system tool provider.

    Parameters
    ----------
    workspace_dir:
        Root directory for all file operations. Defaults to ``.workspace``
        in the current working directory. Created automatically if missing.
    max_file_size:
        Maximum number of bytes allowed for a single ``fs_read`` call.
        Defaults to 1 MB. Pass ``0`` to disable the limit.
    """

    def __init__(
        self,
        workspace_dir: str = _DEFAULT_WORKSPACE,
        max_file_size: int = MAX_FILE_SIZE,
    ) -> None:
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.max_file_size = max_file_size
        os.makedirs(self.workspace_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Path security
    # ------------------------------------------------------------------ #
    def _resolve(self, user_path: str) -> str:
        """Resolve ``user_path`` against ``self.workspace_dir`` and enforce containment.

        Raises ``ValueError`` for empty/None input, ``PermissionError`` for
        absolute paths or paths that escape the workspace.
        """
        if not user_path or not isinstance(user_path, str):
            raise ValueError("路径不能为空")

        # 拒绝绝对路径（在 join 之前直接拦截，不依赖后续 startswith）
        if os.path.isabs(user_path):
            raise PermissionError("不允许使用绝对路径")

        real_workspace = os.path.realpath(self.workspace_dir)
        joined = os.path.join(real_workspace, user_path)
        real_path = os.path.realpath(joined)

        # 确保解析后的路径在 workspace 内
        if real_path != real_workspace and not real_path.startswith(real_workspace + os.sep):
            raise PermissionError(
                f"路径穿越攻击被拦截: '{user_path}' 解析为 '{real_path}'，"
                f"超出工作区 '{real_workspace}'"
            )
        return real_path

    # ------------------------------------------------------------------ #
    # Tool specs
    # ------------------------------------------------------------------ #
    def _read_spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_read",
            description="读取文件内容并返回文本。仅限工作区内路径。",
            parameters=(
                ToolArg(name="path", type="string", description="要读取的文件路径（相对于工作区）", required=True),
            ),
            timeout_seconds=10.0,
        )

    def _write_spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_write",
            description="将文本内容写入文件。仅限工作区内路径。",
            parameters=(
                ToolArg(name="path", type="string", description="目标文件路径（相对于工作区）", required=True),
                ToolArg(name="content", type="string", description="要写入的文本内容", required=True),
            ),
            timeout_seconds=10.0,
        )

    def _list_spec(self) -> ToolSpec:
        return ToolSpec(
            name="fs_list",
            description="列出目录内容。仅限工作区内路径。",
            parameters=(
                ToolArg(name="path", type="string", description="要列出的目录路径（相对于工作区，默认为根目录）", required=False),
            ),
            timeout_seconds=10.0,
        )

    @hookimpl
    def get_tool_specs(self, context: SessionContext) -> tuple[ToolSpec, ...]:
        return (self._read_spec(), self._write_spec(), self._list_spec())

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    @hookimpl
    async def execute_tool(
        self, context: SessionContext, tool: ToolSpec, arguments: dict[str, object]
    ) -> ToolResult | None:
        if tool.name == "fs_read":
            return await self._read(arguments)
        if tool.name == "fs_write":
            return await self._write(arguments)
        if tool.name == "fs_list":
            return await self._list(arguments)
        return None

    async def _read(self, arguments: dict[str, object]) -> ToolResult:
        path = str(arguments.get("path", ""))
        try:
            real_path = self._resolve(path)
            if not os.path.isfile(real_path):
                return ToolResult(
                    tool_name="fs_read",
                    status=ToolResultStatus.ERROR,
                    error=f"文件不存在: '{path}'",
                    output={"path": path},
                )

            file_size = os.path.getsize(real_path)
            if self.max_file_size > 0 and file_size > self.max_file_size:
                return ToolResult(
                    tool_name="fs_read",
                    status=ToolResultStatus.ERROR,
                    error=(
                        f"文件过大 ({file_size / 1024:.0f} KB)，"
                        f"超过 {self.max_file_size / 1024:.0f} KB 限制。"
                    ),
                    output={"path": path, "size": file_size},
                )

            try:
                content = await asyncio.to_thread(self._read_file_sync, real_path)
            except UnicodeDecodeError:
                content = await asyncio.to_thread(
                    lambda: open(real_path, "r", encoding="latin-1").read()
                )
                return ToolResult(
                    tool_name="fs_read",
                    status=ToolResultStatus.OK,
                    output={"content": content, "path": path, "encoding": "latin-1"},
                )
            return ToolResult(tool_name="fs_read", status=ToolResultStatus.OK, output={"content": content, "path": path})
        except PermissionError as exc:
            logger.warning("fs_read blocked: %s", exc)
            return ToolResult(tool_name="fs_read", status=ToolResultStatus.ERROR, error=str(exc), output={"path": path})
        except FileNotFoundError:
            return ToolResult(tool_name="fs_read", status=ToolResultStatus.ERROR, error=f"文件不存在: '{path}'", output={"path": path})
        except Exception as exc:
            logger.exception("fs_read failed")
            return ToolResult(tool_name="fs_read", status=ToolResultStatus.ERROR, error=str(exc), output={"path": path})

    @staticmethod
    def _read_file_sync(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    async def _write(self, arguments: dict[str, object]) -> ToolResult:
        path = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))
        try:
            real_path = self._resolve(path)
            os.makedirs(os.path.dirname(real_path), exist_ok=True)
            with open(real_path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(tool_name="fs_write", status=ToolResultStatus.OK, output={"path": path, "bytes_written": len(content)})
        except PermissionError as exc:
            logger.warning("fs_write blocked: %s", exc)
            return ToolResult(tool_name="fs_write", status=ToolResultStatus.ERROR, error=str(exc), output={"path": path})
        except Exception as exc:
            logger.exception("fs_write failed")
            return ToolResult(tool_name="fs_write", status=ToolResultStatus.ERROR, error=str(exc), output={"path": path})

    async def _list(self, arguments: dict[str, object]) -> ToolResult:
        raw = str(arguments.get("path", "") or "")
        try:
            if raw:
                real_path = self._resolve(raw)
            else:
                real_path = os.path.realpath(self.workspace_dir)
            if not os.path.isdir(real_path):
                return ToolResult(tool_name="fs_list", status=ToolResultStatus.ERROR, error=f"'{raw}' 不是一个目录。", output={"path": raw})
            entries = sorted(os.listdir(real_path))
            files: list[str] = []
            dirs: list[str] = []
            for name in entries:
                full = os.path.join(real_path, name)
                (dirs if os.path.isdir(full) else files).append(name)
            return ToolResult(
                tool_name="fs_list",
                status=ToolResultStatus.OK,
                output={"path": raw or "/", "directories": dirs, "files": files, "count": len(entries)},
            )
        except PermissionError as exc:
            logger.warning("fs_list blocked: %s", exc)
            return ToolResult(tool_name="fs_list", status=ToolResultStatus.ERROR, error=str(exc), output={"path": raw})
        except Exception as exc:
            logger.exception("fs_list failed")
            return ToolResult(tool_name="fs_list", status=ToolResultStatus.ERROR, error=str(exc), output={"path": raw})


__all__ = ["FileSystemPlugin"]

