"""Tests for Phase 6 MCP Client."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyharness import Harness
from pyharness.core import _settle
from pyharness.plugins.mcp_client import MCPClientPlugin
from pyharness.plugins.mcp_config import load_mcp_config
from pyharness.plugins.mcp_transport import SSETransport, StdioTransport
from pyharness.schema import MCPServerConfig, MCPToolMapping, MCPToolResult, ToolResultStatus
from pyharness.schema import HarnessConfig


def _harness(*plugins, auto_load: bool = False) -> Harness:
    h = Harness(config=HarnessConfig(auto_load_entry_points=auto_load))
    for plugin in plugins:
        h.register_plugin(plugin)
    h.initialize()
    return h


# ---------------------------------------------------------------------------
# 1. Config parsing
# ---------------------------------------------------------------------------
class TestMCPConfig:
    def test_load_default_config(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "mcpServers": {
                        "fs": {"command": "npx", "args": ["-y", "server-fs"], "enabled": True},
                        "remote": {"url": "http://localhost:8080/sse", "transport": "sse", "enabled": True},
                    }
                },
                f,
            )
            path = f.name
        try:
            configs = load_mcp_config(path)
            assert len(configs) == 2
            assert configs[0].name == "fs"
            assert configs[0].transport == "stdio"
            assert configs[0].command == "npx"
            assert configs[1].name == "remote"
            assert configs[1].transport == "sse"
            assert configs[1].url == "http://localhost:8080/sse"
        finally:
            os.unlink(path)

    def test_missing_config_returns_empty(self) -> None:
        configs = load_mcp_config("/nonexistent/path/mcp_servers.json")
        assert configs == []

    def test_disabled_server_excluded(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "mcpServers": {
                        "disabled": {"command": "npx", "args": ["-y", "x"], "enabled": False},
                    }
                },
                f,
            )
            path = f.name
        try:
            configs = load_mcp_config(path)
            assert len(configs) == 1
            assert configs[0].enabled is False
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 2. StdioTransport lifecycle
# ---------------------------------------------------------------------------
class TestStdioTransport:
    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        transport = StdioTransport("python", ["-c", "import time; time.sleep(0.1)"])
        await transport.connect()
        assert transport.process is not None
        await transport.disconnect()
        assert transport.process is None

    @pytest.mark.asyncio
    async def test_send_request_timeout(self) -> None:
        transport = StdioTransport("python", ["-c", "import time; time.sleep(5)"])
        await transport.connect()
        with pytest.raises(TimeoutError):
            await transport.send_request("test", timeout=0.5)
        await transport.disconnect()

    @pytest.mark.asyncio
    async def test_pending_request_fails_when_subprocess_dies(self) -> None:
        """If the child process exits while a request is in-flight, the
        pending future must be failed with ``ConnectionError`` immediately
        instead of waiting for the 30s default timeout."""
        # Script that reads one line then exits (no response).
        transport = StdioTransport(
            "python", ["-c", "import sys; sys.stdin.readline(); raise SystemExit(7)"]
        )
        await transport.connect()
        try:
            # Bound the wait generously above the 30s default to prove the
            # reader's finally-clause short-circuits the wait.
            start = time.monotonic()
            with pytest.raises(ConnectionError):
                await transport.send_request("ping", timeout=30.0)
            elapsed = time.monotonic() - start
            assert elapsed < 5.0, f"send_request waited {elapsed:.1f}s (expected <5s)"
            # After the failure, the transport must have cleared its pending map.
            assert transport._pending == {}
        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_stderr_drain_task_is_running(self) -> None:
        """connect() must spawn a stderr drain task to avoid deadlocking the
        child on a full stderr pipe."""
        transport = StdioTransport("python", ["-c", "import time; time.sleep(0.2)"])
        await transport.connect()
        try:
            assert transport._stderr_task is not None
            assert not transport._stderr_task.done()
        finally:
            await transport.disconnect()
        # disconnect must cancel the drain task.
        assert transport._stderr_task is None or transport._stderr_task.done()


# ---------------------------------------------------------------------------
# 3. SSETransport basic
# ---------------------------------------------------------------------------
class TestSSETransport:
    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        transport = SSETransport("http://localhost:9999")
        await transport.connect()
        assert transport._client is not None
        await transport.disconnect()
        assert transport._client is None


# ---------------------------------------------------------------------------
# 4. MCP tool discovery
# ---------------------------------------------------------------------------
class TestMCPToolDiscovery:
    @pytest.mark.asyncio
    async def test_tool_discovery(self) -> None:
        plugin = MCPClientPlugin(config_path="mcp_servers.json")

        mock_transport = MagicMock()
        mock_transport.connect = AsyncMock(return_value=None)
        mock_transport.send_request = AsyncMock(return_value={
            "tools": [
                {"name": "read_file", "description": "Read a file", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
            ]
        })
        mock_transport.disconnect = AsyncMock(return_value=None)

        config = MCPServerConfig(name="test-server", command="echo", args=["ok"], transport="stdio")

        with patch("pyharness.plugins.mcp_client.StdioTransport", return_value=mock_transport):
            await plugin._connect_server(config)

        assert "mcp_test-server_read_file" in plugin.tool_mappings
        mapping = plugin.tool_mappings["mcp_test-server_read_file"]
        assert mapping.tool_name == "read_file"
        assert mapping.server_name == "test-server"


# ---------------------------------------------------------------------------
# 5. MCP tool call routing
# ---------------------------------------------------------------------------
class TestMCPToolCall:
    @pytest.mark.asyncio
    async def test_tool_call_routed(self) -> None:
        plugin = MCPClientPlugin()
        plugin.harness = _harness()

        mock_transport = MagicMock()
        mock_transport.send_request = AsyncMock(return_value={
            "content": [{"type": "text", "text": "file content"}],
            "isError": False,
        })
        plugin.connections["test-server"] = mock_transport
        plugin.tool_mappings["mcp_test-server_read_file"] = MCPToolMapping(
            server_name="test-server",
            tool_name="read_file",
            harness_name="mcp_test-server_read_file",
            description="Read a file",
            input_schema={},
        )

        mock_tool = MagicMock()
        mock_tool.name = "mcp_test-server_read_file"
        result = await plugin.execute_tool(None, mock_tool, {"path": "/tmp/test.txt"})
        assert result is not None
        assert result.status == ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_tool_not_found(self) -> None:
        plugin = MCPClientPlugin()
        mock_tool = MagicMock()
        mock_tool.name = "mcp_unknown_tool"
        result = await plugin.execute_tool(None, mock_tool, {})
        assert result is not None
        assert result.status == ToolResultStatus.ERROR
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_server_not_connected(self) -> None:
        plugin = MCPClientPlugin()
        plugin.tool_mappings["mcp_test_tool"] = MCPToolMapping(
            server_name="disconnected-server",
            tool_name="tool",
            harness_name="mcp_test_tool",
        )
        mock_tool = MagicMock()
        mock_tool.name = "mcp_test_tool"
        result = await plugin.execute_tool(None, mock_tool, {})
        assert result is not None
        assert result.status == ToolResultStatus.ERROR
        assert "not connected" in result.error


# ---------------------------------------------------------------------------
# 6. Server disconnect
# ---------------------------------------------------------------------------
class TestMCPServerDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_removes_mappings(self) -> None:
        plugin = MCPClientPlugin()
        mock_transport = MagicMock()
        mock_transport.disconnect = AsyncMock(return_value=None)
        plugin.connections["test-server"] = mock_transport
        plugin.tool_mappings["mcp_test_tool"] = MCPToolMapping(
            server_name="test-server",
            tool_name="tool",
            harness_name="mcp_test_tool",
        )

        await plugin.mcp_disconnect("test-server")
        assert "test-server" not in plugin.connections
        assert "mcp_test_tool" not in plugin.tool_mappings


__all__ = [
    "TestMCPConfig",
    "TestStdioTransport",
    "TestSSETransport",
    "TestMCPToolDiscovery",
    "TestMCPToolCall",
    "TestMCPServerDisconnect",
]
