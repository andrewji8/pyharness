"""Guard / Approval plugin for PyHarness.

Intercepts high-risk tool calls and requests human confirmation before
allowing them to execute. Uses the ``pre_tool_execution`` hook to short-
circuit execution, and ``ask_user_confirmation`` to delegate the actual
prompt to whichever UI plugin is active (CLI, Web, etc.).

Design
------
* **Blacklist-driven** — a configurable set of tool names are treated as
  high-risk and require confirmation.
* **Decoupled UI** — the plugin does not call ``input()`` directly. It calls
  the ``ask_user_confirmation`` hook and lets a UI plugin present the dialog.
* **Graceful degradation** — when no UI plugin is registered, the plugin
  falls back to a configurable default action (``"reject"`` or ``"allow"``).
* **No pm in constructor** — the plugin manager reference is captured via
  the ``harness_initialized`` hook, so callers never pass ``pm`` explicitly.
* **Session trust** — once a user approves a tool and opts in to "trust for
  this session", the tool is added to ``trusted_tools`` and subsequent calls
  bypass confirmation. This avoids repeated prompts when the LLM issues
  parallel tool calls.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import ToolCall, ToolResult, ToolResultStatus

if TYPE_CHECKING:
    from pyharness.core import Harness

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")

# Default high-risk tool names. Plugins can override via constructor.
_DEFAULT_BLACKLIST = frozenset({"python_exec", "shell_exec", "fs_write", "fs_delete"})


class ApprovalGuardPlugin:
    """Pre-tool-execution guard that asks for human approval on risky tools.

    Parameters
    ----------
    blacklist:
        Tool names that require confirmation. Defaults to a conservative set
        of file-system and code-execution tools.
    default_action:
        What to do when no UI plugin is available to ask the user.
        ``"reject"`` (default) blocks the tool; ``"allow"`` lets it through.
    """

    def __init__(
        self,
        blacklist: frozenset[str] | set[str] | None = None,
        default_action: str = "reject",
    ) -> None:
        self.blacklist = frozenset(blacklist) if blacklist is not None else _DEFAULT_BLACKLIST
        if default_action not in {"reject", "allow"}:
            raise ValueError(f"default_action must be 'reject' or 'allow', got {default_action!r}")
        self.default_action = default_action
        self.pm: Any = None
        # Tool names the user has explicitly trusted for the current session.
        self.trusted_tools: set[str] = set()

    # ------------------------------------------------------------------ #
    # Lifecycle: capture the plugin manager and reset per-session state.
    # ------------------------------------------------------------------ #
    @hookimpl
    def harness_initialized(self, harness: "Harness") -> None:
        """Capture the pluggy manager and reset session-local trust set."""
        self.pm = harness.bus.pm
        self.trusted_tools.clear()

    # ------------------------------------------------------------------ #
    # Hook implementation
    # ------------------------------------------------------------------ #
    @hookimpl(tryfirst=True)
    async def pre_tool_execution(self, tool_call: ToolCall) -> ToolResult | None:
        """Intercept ``tool_call`` if its name is in the blacklist.

        Returns a ``ToolResult`` carrying a clear error message when the user
        (or fallback policy) declines, or ``None`` to allow the tool to
        execute normally.
        """
        tool_name = tool_call.tool_name

        # 1. 已信任的工具直接放行。
        if tool_name in self.trusted_tools:
            return None

        # 2. 不在黑名单的工具直接放行。
        if tool_name not in self.blacklist:
            return None

        # 3. 高危工具：请求用户审批。
        prompt = (
            f"⚠️ Agent 请求执行高危工具: {tool_name}\n"
            f"参数: {tool_call.arguments}\n是否允许？"
        )
        metadata = {
            "tool_name": tool_name,
            "arguments": tool_call.arguments,
            "tool_call_id": tool_call.id,
        }

        allowed = await self._ask_user(prompt, metadata)
        if not allowed:
            logger.warning("Guard: user rejected %s", tool_name)
            return self._default_reject(tool_call)

        # 4. 用户同意：询问是否信任本次会话（批量审批）。
        trusted = await self._ask_trust_session(tool_name)
        if trusted:
            self.trusted_tools.add(tool_name)
            logger.info("Guard: user trusted %s for this session", tool_name)
        else:
            logger.info("Guard: user approved %s (one-time only)", tool_name)

        return None

    # ------------------------------------------------------------------ #
    # Confirmation flow
    # ------------------------------------------------------------------ #
    async def _ask_user(self, prompt: str, metadata: dict[str, Any]) -> bool:
        """Ask the user for confirmation via the ``ask_user_confirmation`` hook.

        If any plugin implements the hook and returns a non-None value, that
        value is used. If the hook is not implemented (empty return list) or
        raises, fall back to ``self.default_action``.
        """
        try:
            hook_result = self.pm.hook.ask_user_confirmation(prompt=prompt, metadata=metadata)
        except Exception:
            logger.debug("ask_user_confirmation hook failed; using fallback.")
            return self.default_action == "allow"

        # pluggy may return a single awaitable or an iterable of awaitables;
        # normalize into a list and check whether any plugin actually answered.
        if inspect.isawaitable(hook_result):
            candidates = [hook_result]
        else:
            candidates = list(hook_result)

        if not candidates:
            return self.default_action == "allow"

        for r in candidates:
            if inspect.isawaitable(r):
                try:
                    r = await r
                except Exception:
                    continue
            if r is not None:
                return bool(r)

        return self.default_action == "allow"

    async def _ask_trust_session(self, tool_name: str) -> bool:
        """Ask the user whether to trust ``tool_name`` for the rest of the session.

        Returns ``True`` if the user opted in to trust this tool, ``False``
        otherwise (including when no UI plugin is present or the hook fails).
        """
        prompt = (
            f"是否在本次会话中信任工具 '{tool_name}'？\n"
            f"（信任后，后续调用 '{tool_name}' 将不再询问，直接放行）"
        )
        metadata = {"tool_name": tool_name, "trust_session": True}
        try:
            hook_result = self.pm.hook.ask_user_confirmation(prompt=prompt, metadata=metadata)
        except Exception:
            return False

        if inspect.isawaitable(hook_result):
            candidates = [hook_result]
        else:
            candidates = list(hook_result)

        if not candidates:
            return False

        for r in candidates:
            if inspect.isawaitable(r):
                try:
                    r = await r
                except Exception:
                    continue
            if r is not None:
                return bool(r)

        return False

    # ------------------------------------------------------------------ #
    # Rejection helper
    # ------------------------------------------------------------------ #
    def _default_reject(self, tool_call: ToolCall) -> ToolResult:
        """Build a clear, LLM-friendly rejection ToolResult."""
        return ToolResult(
            tool_name=tool_call.tool_name,
            status=ToolResultStatus.ERROR,
            error=(
                f"⚠️ 用户拒绝了工具 '{tool_call.tool_name}' 的执行。"
                f"请不要重试此操作，尝试用其他方式完成任务，"
                f"或向用户解释你需要执行此操作的原因。"
            ),
            output={
                "rejected": True,
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
            },
        )


__all__ = ["ApprovalGuardPlugin"]
