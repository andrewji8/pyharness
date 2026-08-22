"""MCP transport layer abstractions.

Provides:
- ``StdioTransport``: communicate with an MCP Server over stdio (subprocess).
- ``SSETransport``: communicate with an MCP Server over SSE / HTTP.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from pyharness.schema import MCPServerConfig

logger = logging.getLogger(__name__)


class StdioTransport:
    """stdio transport: launch a subprocess and exchange JSON-RPC over stdin/stdout."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        self.command = command
        self.args = args
        self.env = env or {}
        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Start the subprocess and begin reading responses."""
        import os

        full_env = {**os.environ, **self.env}
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def send_request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
        """Send a JSON-RPC request and return the result payload."""
        if self.process is None or self.process.stdin is None:
            raise ConnectionError("StdioTransport is not connected")

        self._request_id += 1
        req_id = self._request_id
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params:
            payload["params"] = params

        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        line = json.dumps(payload) + "\n"
        self.process.stdin.write(line.encode())
        await self.process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"MCP request '{method}' timed out after {timeout}s")

    async def _read_loop(self) -> None:
        """Background task: read JSON-RPC responses from stdout."""
        assert self.process and self.process.stdout
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode().strip())
                req_id = msg.get("id")
                if req_id is not None and req_id in self._pending:
                    future = self._pending.pop(req_id)
                    if "result" in msg:
                        future.set_result(msg["result"])
                    elif "error" in msg:
                        future.set_exception(RuntimeError(f"MCP error: {msg['error']}"))
                    else:
                        future.set_result({})
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception as exc:
                logger.debug("StdioTransport reader error: %s", exc)

    async def disconnect(self) -> None:
        """Terminate the subprocess and clean up."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None
        self._pending.clear()


class SSETransport:
    """SSE transport: send JSON-RPC requests over HTTP POST."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._request_id = 0

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request via HTTP POST and return the result."""
        if self._client is None:
            raise ConnectionError("SSETransport is not connected")

        self._request_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self._request_id, "method": method}
        if params:
            payload["params"] = params

        response = await self._client.post(self.url, json=payload)
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(f"MCP error: {body['error']}")
        return body.get("result", {})

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["StdioTransport", "SSETransport"]
