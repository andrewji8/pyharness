"""Context Compaction Plugin.

Sliding-window + summary compaction to prevent token explosions in long
agent sessions. Implements the ``build_request`` hook so that, whenever the
assembled message list exceeds a token budget, older turns are replaced by
a single LLM-generated summary message.

Design
------
* **System-prompt preservation** — ``messages[0]`` is always kept intact.
* **Recent-turn preservation** — the last ``KEEP_RECENT_TURNS`` user/assistant
  pairs are kept verbatim so the model retains immediate context.
* **Lightweight token estimation** — prefers a pure-Python ``len(text)//3``
  heuristic; optionally uses ``tiktoken`` if available.
* **Graceful degradation** — if summary generation fails, falls back to a
  simple truncation strategy so the main loop never blocks.

Usage
-----
::

    from pyharness.plugins.context_compaction import ContextCompactionPlugin
    from pyharness import Harness

    h = Harness()
    h.bus.register(ContextCompactionPlugin(max_tokens=8000, keep_recent_turns=3))
    h.initialize()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import Message, Role

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")

# Default tuning constants (overridable via constructor).
KEEP_RECENT_TURNS: int = 3
MAX_TOKENS: int = 8000


def _estimate_tokens(text: str) -> int:
    """Estimate token count for ``text`` without heavy C-extension dependencies.

    Tries ``tiktoken`` (``cl100k_base``) first; falls back to the lightweight
    ``len(text) // 3`` heuristic when ``tiktoken`` is unavailable or its
    encoder initialization fails.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 3)


def _count_messages_tokens(messages: list[Message]) -> int:
    """Total estimated tokens across all message contents."""
    return sum(_estimate_tokens(m.content) for m in messages)


def _truncate_messages(messages: list[Message], max_tokens: int) -> list[Message]:
    """Fallback compaction: keep the system prompt plus as many recent messages
    as fit within ``max_tokens`` (greedy, oldest-first eviction)."""
    if not messages:
        return messages

    kept: list[Message] = [messages[0]]
    budget = max_tokens - _estimate_tokens(messages[0].content)
    if budget <= 0:
        return kept

    for msg in reversed(messages[1:]):
        cost = _estimate_tokens(msg.content)
        if cost > budget:
            break
        kept.insert(1, msg)
        budget -= cost

    return kept


class ContextCompactionPlugin:
    """Sliding-window context compaction via the ``build_request`` hook.

    Parameters
    ----------
    max_tokens:
        Token budget that triggers compaction when the assembled message list
        exceeds it.
    keep_recent_turns:
        Number of recent user/assistant pairs (each pair = 2 messages) to
        preserve verbatim.
    summarizer:
        Optional async-compatible callable ``(text: str) -> str`` used to
        generate a history summary. When ``None``, the plugin falls back to
        a deterministic truncation heuristic.
    """

    def __init__(
        self,
        max_tokens: int = MAX_TOKENS,
        keep_recent_turns: int = KEEP_RECENT_TURNS,
        summarizer: Callable[[str], Any] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.keep_recent_turns = keep_recent_turns
        self._summarizer = summarizer

    # ------------------------------------------------------------------ #
    # Hook implementation
    # ------------------------------------------------------------------ #
    @hookimpl(trylast=True)
    async def build_request(self, messages: list[Message]) -> list[Message]:
        """Compact ``messages`` when the estimated token count exceeds budget.

        Returns the original list unchanged when compaction is unnecessary
        or when summary generation fails (fallback: truncation).
        """
        if not messages:
            return messages

        total = _count_messages_tokens(messages)
        if total <= self.max_tokens:
            return messages

        logger.info(
            "ContextCompaction: %d tokens exceed budget %d; compacting.",
            total,
            self.max_tokens,
        )

        # 1. Always keep the first message (system prompt) intact.
        system_msg = messages[0]

        # 2. Keep the last KEEP_RECENT_TURNS pairs (user+assistant) verbatim.
        keep_count = self.keep_recent_turns * 2
        recent: list[Message] = messages[-keep_count:] if len(messages) > 1 else []
        if not recent:
            return messages

        # 3. The middle slice is the history to summarize.
        history: list[Message] = messages[1:-keep_count]
        if not history:
            return messages

        try:
            summary_text = await self._generate_summary(history)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Summary generation failed (%s); falling back to truncation.", exc
            )
            return _truncate_messages(messages, self.max_tokens)

        summary_msg = Message(
            role=Role.SYSTEM,
            content=f"[历史对话摘要]: {summary_text}",
        )
        return [system_msg, summary_msg, *recent]

    # ------------------------------------------------------------------ #
    # Summary generation
    # ------------------------------------------------------------------ #
    async def _generate_summary(self, history: list[Message]) -> str:
        """Produce a concise summary of ``history`` messages.

        If a ``summarizer`` callable was provided at construction time, it is
        invoked with the concatenated history text. Otherwise a lightweight
        heuristic is used as a placeholder.
        """
        if self._summarizer is not None:
            text = "\n".join(f"{m.role}: {m.content}" for m in history)
            result = self._summarizer(text)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)

        # --- Placeholder heuristic ---------------------------------------------------
        # TODO: Replace with a real LLM call via the existing LLM provider plugin.
        #
        #   from pyharness.plugins.llm.entry import _registry
        #   from pyharness.schema import LLMRequest
        #
        #   text = "\n".join(f"{m.role}: {m.content}" for m in history)
        #   request = LLMRequest(
        #       model="<summary-model>",
        #       messages=[
        #           Message(role=Role.SYSTEM, content="Summarize the following history concisely."),
        #           Message(role=Role.USER, content=text),
        #       ],
        #   )
        #   response = await _registry.chat(request)
        #   return response.content if response else text[:2000]
        # ------------------------------------------------------------------------------

        text = "\n".join(f"{m.role}: {m.content}" for m in history)
        return text[:2000] + ("…[truncated]" if len(text) > 2000 else "")


__all__ = ["ContextCompactionPlugin", "KEEP_RECENT_TURNS", "MAX_TOKENS"]
