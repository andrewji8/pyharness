"""CLI UI plugin for PyHarness human-in-the-loop interactions.

Implements the ``ask_user_confirmation`` hook so the Guard plugin (or any
other policy plugin) can prompt the user for approval in an async-safe way.

Key technique
-------------
``input()`` is a blocking call. Calling it directly from an ``async def``
would freeze the entire event loop. We avoid that by offloading the blocking
read to a thread pool via ``asyncio.to_thread(...)`` (Python 3.9+). The
event loop stays responsive while the terminal waits for the user.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import Event

hookimpl = HookimplMarker("pyharness")
logger = logging.getLogger(__name__)


class CLIUIPlugin:
    """Render human-in-the-loop prompts in the terminal.

    Register this plugin with the harness so that ``ask_user_confirmation``
    hooks route to the CLI. In headless environments (tests, CI) simply omit
    this plugin and the Guard will fall back to its configured default.
    """

    DEFAULT_TIMEOUT: float = 60.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    @hookimpl
    async def ask_user_confirmation(self, prompt: str, metadata: dict[str, Any]) -> bool | None:
        """Display a warning and wait for ``y/N`` in a thread.

        Returns ``True`` for ``y``/``yes``, ``False`` for anything else
        (including timeout, EOF, or ``Ctrl-C``).
        """
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(self._blocking_prompt, prompt),
                timeout=self.timeout,
            )
            return answer.strip().lower() in {"y", "yes"}
        except asyncio.TimeoutError:
            print(f"\n⏰ 确认超时（{self.timeout:.0f}s），默认拒绝。")
            return False
        except (KeyboardInterrupt, EOFError):
            return False
        except Exception as exc:
            logger.debug("CLI confirmation failed (%s); returning False.", exc)
            return False

    @staticmethod
    def _blocking_prompt(prompt: str) -> str:
        """Run in a worker thread so the event loop is not blocked."""
        try:
            # ANSI red bold + reset for visibility.
            print(f"\033[31m\033[1m⚠️  GUARD\033[0m {prompt}")
            print("   \033[1mAllow this operation? [y/N]:\033[0m ", end="", flush=True)
            return input().strip()
        except EOFError:
            return "n"


__all__ = ["CLIUIPlugin"]
