"""MCP client — JSON-RPC 2.0 over stdio or HTTP transport."""

from __future__ import annotations

import itertools
import json
import logging
from typing import Any

from eyetor.mcp.transport import BaseTransport, HttpTransport, StdioTransport
from eyetor.models.tools import ToolDefinition

logger = logging.getLogger(__name__)

_id_counter = itertools.count(1)


def _next_id() -> int:
    return next(_id_counter)


class McpClient:
    """MCP client that connects to a single MCP server.

    Supports both stdio and HTTP transports. Discovered tools are
    converted to ToolDefinition objects for use with ToolRegistry.
    """

    def __init__(self, transport: BaseTransport) -> None:
        self._transport = transport
        self._tools: list[ToolDefinition] = []

    async def connect(self) -> None:
        """Start transport and perform the MCP initialization handshake."""
        await self._transport.start()
        await self._initialize()
        await self._discover_tools()

    async def _rpc(self, method: str, params: dict | None = None) -> Any:
        """Send a JSON-RPC request and return the result."""
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": _next_id(),
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        if isinstance(self._transport, HttpTransport):
            response = await self._transport.send(msg)
        else:
            await self._transport.send(msg)
            response = await self._transport.receive()

        if "error" in response:
            raise RuntimeError(f"MCP RPC error: {response['error']}")
        return response.get("result")

    async def _initialize(self) -> None:
        await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "eyetor", "version": "0.1.0"},
            },
        )
        logger.debug("MCP server initialized")

    async def _discover_tools(self) -> None:
        result = await self._rpc("tools/list")
        raw_tools = result.get("tools", []) if result else []
        self._tools = [_mcp_tool_to_definition(t, self) for t in raw_tools]
        logger.debug("MCP tools discovered: %d", len(self._tools))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server and return the result as string.

        Retries once on connection failure, attempting to reconnect for stdio
        transports (the subprocess may have died).
        """
        for attempt in range(2):
            try:
                result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
                if not result:
                    return ""
                content = result.get("content", [])
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts)
            except (EOFError, RuntimeError, OSError) as exc:
                if attempt == 0:
                    logger.warning("MCP call_tool '%s' failed (%s), reconnecting...", name, exc)
                    try:
                        await self._transport.close()
                    except Exception:
                        pass
                    await self.connect()
                else:
                    raise
        return ""  # unreachable, keeps mypy happy

    def get_tools(self) -> list[ToolDefinition]:
        """Return tool definitions for all discovered MCP tools."""
        return list(self._tools)

    async def close(self) -> None:
        await self._transport.close()


def _mcp_tool_to_definition(raw: dict, client: McpClient) -> ToolDefinition:
    """Convert an MCP tools/list entry to a ToolDefinition."""
    name = raw["name"]
    description = raw.get("description", "")
    input_schema = raw.get("inputSchema", {"type": "object", "properties": {}})

    async def handler(**kwargs: Any) -> str:
        return await client.call_tool(name, kwargs)

    return ToolDefinition(
        name=name,
        description=description,
        parameters=input_schema,
        handler=handler,
    )
