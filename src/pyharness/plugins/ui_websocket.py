"""WebSocket Observer 插件：监听 ``observe`` hook，将事件转发给 WebUI。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import Event
from pyharness.specs import AgentHooks

hookimpl = HookimplMarker("pyharness")

if TYPE_CHECKING:
    from pyharness.plugins.web_ui import WebUIPlugin

logger = logging.getLogger(__name__)


class WebSocketObserverPlugin:
    """监听 ``observe`` hook，将事件转发给 WebUI 的 WebSocket 客户端。"""

    def __init__(self, web_ui: WebUIPlugin) -> None:
        self.web_ui = web_ui

    @hookimpl
    async def observe(self, context: SessionContext, event: Event) -> None:
        """接收引擎广播的事件，推送给前端。"""
        if event.type.startswith("_"):
            return
        await self.web_ui.broadcast(event.type, {
            "session_id": str(event.session_id),
            **event.payload,
        })


__all__ = ["WebSocketObserverPlugin"]
