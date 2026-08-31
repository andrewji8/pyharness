"""MCP Client plugin for PyHarness.

Manages connections to multiple MCP Servers, discovers their tools, and routes
PyHarness tool calls to the appropriate server.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pluggy import HookimplMarker

from pyharness.core import _settle
from pyharness.plugins.mcp_config import load_mcp_config
from pyharness.plugins.mcp_transport import SSETransport, StdioTransport
from pyharness.schema import MCPServerConfig, MCPToolMapping, MCPToolResult, ToolArg, ToolResult, ToolResultStatus, ToolSpec
from pyharness.specs import AgentHooks

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")


class MCPClientPlugin:
    """MCP Client plugin: manages multiple MCP Server connections."""

    def __init__(self, config_path: str = "mcp_servers.json") -> None:
        self.config_path = config_path
        self.connections: dict[str, StdioTransport | SSETransport] = {}
        self.tool_mappings: dict[str, MCPToolMapping] = {}
        self.harness: Any = None

    @hookimpl
    def harness_initialized(self, harness: Any) -> None:
        """Capture harness reference and auto-connect enabled servers."""
        self.harness = harness
        asyncio.get_event_loop().create_task(self._auto_connect())

    async def _auto_connect(self) -> None:
        """Connect to all enabled servers from the config file."""
        try:
            configs = load_mcp_config(self.config_path)
        except Exception as exc:
            logger.warning("Failed to load MCP config '%s': %s", self.config_path, exc)
            return

        for config in configs:
            if not config.enabled:
                continue
            try:
                await self._connect_server(config)
            except Exception as exc:
                logger.warning("MCP Server '%s' connection failed: %s", config.name, exc)

    async def _connect_server(self, config: MCPServerConfig) -> None:
        """Connect to a single MCP Server and discover its tools."""
        if config.transport == "stdio":
            if not config.command:
                raise ValueError("stdio transport requires 'command'")
            transport = StdioTransport(config.command, config.args, config.env)
        else:
            if not config.url:
                raise ValueError("sse transport requires 'url'")
            transport = SSETransport(config.url)

        await transport.connect()

        try:
            init_result = await transport.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "PyHarness", "version": "0.5.0"},
                },
            )
            logger.debug("MCP Server '%s' initialized: %s", config.name, init_result)

            tools_result = await transport.send_request("tools/list")
            tools = tools_result.get("tools", [])

            for tool in tools:
                harness_name = f"mcp_{config.name}_{tool['name']}"
                mapping = MCPToolMapping(
                    server_name=config.name,
                    tool_name=tool["name"],
                    harness_name=harness_name,
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                )
                self.tool_mappings[harness_name] = mapping

            self.connections[config.name] = transport
            logger.info("MCP Server '%s' connected with %d tools", config.name, len(tools))
        except Exception:
            await transport.disconnect()
            raise

    @hookimpl
    def get_tool_specs(self, context: Any) -> tuple[ToolSpec, ...]:
        """Expose all discovered MCP tools as PyHarness ToolSpecs."""
        specs = []
        for harness_name, mapping in self.tool_mappings.items():
            args = self._schema_to_tool_args(mapping.input_schema)
            specs.append(
                ToolSpec(
                    name=harness_name,
                    description=f"[MCP:{mapping.server_name}] {mapping.description}",
                    parameters=tuple(args),
                )
            )
        return tuple(specs)

    @hookimpl
    async def execute_tool(self, context: Any, tool: ToolSpec, arguments: dict[str, object]) -> ToolResult | None:
        """Route MCP tool calls to the appropriate server."""
        if not tool.name.startswith("mcp_"):
            return None

        mapping = self.tool_mappings.get(tool.name)
        if mapping is None:
            return ToolResult(
                tool_name=tool.name,
                status=ToolResultStatus.ERROR,
                error=f"MCP tool '{tool.name}' not found",
            )

        transport = self.connections.get(mapping.server_name)
        if transport is None:
            return ToolResult(
                tool_name=tool.name,
                status=ToolResultStatus.ERROR,
                error=f"MCP Server '{mapping.server_name}' is not connected",
            )

        try:
            result = await transport.send_request(
                "tools/call",
                {"name": mapping.tool_name, "arguments": arguments},
            )
            # TODO: Bridge MCP streaming to tool_emitter when SDK supports
            # streaming callbacks. When the underlying MCP SDK provides a
            # streaming transport (e.g. MCPProtocol.streamingCallTool),
            # forward each chunk through context.tool_emitter as
            # ToolStreamEvent(tool_name=tool.name, stream_type="log", content=chunk).
            mcp_result = MCPToolResult(
                content=result.get("content", []),
                is_error=result.get("isError", False),
            )
            if mcp_result.is_error:
                return ToolResult(
                    tool_name=tool.name,
                    status=ToolResultStatus.ERROR,
                    error=mcp_result.to_text(),
                )
            return ToolResult(
                tool_name=tool.name,
                status=ToolResultStatus.OK,
                output={"result": mcp_result.to_text()},
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool.name,
                status=ToolResultStatus.ERROR,
                error=f"MCP call failed: {type(exc).__name__}: {exc}",
            )

    @hookimpl
    async def mcp_connect(self, config: MCPServerConfig) -> bool:
        """Connect to an MCP Server dynamically."""
        if config.name in self.connections:
            logger.info("MCP Server '%s' already connected", config.name)
            return True
        try:
            await self._connect_server(config)
            return True
        except Exception as exc:
            logger.error("Failed to connect MCP Server '%s': %s", config.name, exc)
            return False

    @hookimpl
    async def mcp_disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP Server."""
        transport = self.connections.pop(server_name, None)
        if transport is not None:
            await transport.disconnect()
            logger.info("MCP Server '%s' disconnected", server_name)

        self.tool_mappings = {k: v for k, v in self.tool_mappings.items() if v.server_name != server_name}

    @hookimpl
    async def mcp_list_tools(self, server_name: str) -> list[dict]:
        """List tools for a connected MCP Server."""
        mappings = [m for m in self.tool_mappings.values() if m.server_name == server_name]
        return [
            {
                "name": m.tool_name,
                "description": m.description,
                "inputSchema": m.input_schema,
                "harness_name": m.harness_name,
            }
            for m in mappings
        ]

    @hookimpl
    async def mcp_call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """Call a tool on a connected MCP Server."""
        transport = self.connections.get(server_name)
        if transport is None:
            return {"error": f"Server '{server_name}' not connected"}

        try:
            return await transport.send_request("tools/call", {"name": tool_name, "arguments": arguments})
        except Exception as exc:
            return {"error": str(exc)}

    @hookimpl
    async def mcp_list_servers(self) -> list[dict]:
        """List all configured/connected MCP Servers."""
        result = []
        try:
            configs = load_mcp_config(self.config_path)
        except Exception:
            configs = []

        for config in configs:
            connected = config.name in self.connections
            tool_count = sum(1 for m in self.tool_mappings.values() if m.server_name == config.name)
            result.append(
                {
                    "name": config.name,
                    "transport": config.transport,
                    "enabled": config.enabled,
                    "connected": connected,
                    "tool_count": tool_count,
                }
            )
        return result

    @staticmethod
    def _schema_to_tool_args(schema: dict[str, Any]) -> list[ToolArg]:
        """Convert a JSON Schema into PyHarness ToolArg list."""
        args: list[ToolArg] = []
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for name, prop in properties.items():
            arg_type = "string"
            if isinstance(prop, dict):
                ptype = prop.get("type", "string")
                if ptype == "integer":
                    arg_type = "integer"
                elif ptype == "number":
                    arg_type = "number"
                elif ptype == "boolean":
                    arg_type = "boolean"
                elif ptype == "array":
                    arg_type = "array"
                elif ptype == "object":
                    arg_type = "object"

            args.append(
                ToolArg(
                    name=name,
                    type=arg_type,
                    description=prop.get("description", ""),
                    required=name in required,
                )
            )
        return args


__all__ = ["MCPClientPlugin"]
