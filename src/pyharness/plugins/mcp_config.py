"""MCP configuration loader.

Parses ``mcp_servers.json`` (Claude Desktop compatible format) into
:class:`MCPServerConfig` objects.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pyharness.schema import MCPServerConfig


def load_mcp_config(path: str = "mcp_servers.json") -> list[MCPServerConfig]:
    """Load MCP server configurations from a JSON file.

    Supports the Claude Desktop ``mcpServers`` format::

        {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
                },
                "remote": {
                    "url": "http://localhost:8080/sse",
                    "transport": "sse"
                }
            }
        }

    Returns an empty list if the file does not exist or is malformed.
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: Any = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to parse MCP config '{path}': {exc}") from exc

    servers = raw.get("mcpServers", {})
    if not isinstance(servers, dict):
        return []

    configs: list[MCPServerConfig] = []
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        transport = "sse" if "url" in cfg else "stdio"
        configs.append(
            MCPServerConfig(
                name=name,
                command=cfg.get("command"),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                url=cfg.get("url"),
                transport=transport,  # type: ignore[arg-type]
                enabled=cfg.get("enabled", True),
            )
        )
    return configs


__all__ = ["load_mcp_config"]
