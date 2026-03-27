"""MCP server registry — manages multiple MCP server connections."""

from __future__ import annotations

import logging
from typing import Any

from eyetor.config import McpServerConfig
from eyetor.mcp.client import McpClient
from eyetor.mcp.transport import HttpTransport, StdioTransport
from eyetor.models.tools import ToolRegistry

logger = logging.getLogger(__name__)


class McpRegistry:
    """Manages connections to multiple MCP servers.

    Usage:
        registry = McpRegistry(config.mcp_servers)
        await registry.connect_all()
        registry.register_all_into(tool_registry)
        # ... use the agent ...
        await registry.close_all()
    """

    def __init__(self, servers_config: dict[str, McpServerConfig]) -> None:
        self._config = servers_config
        self._clients: dict[str, McpClient] = {}

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        for name, cfg in self._config.items():
            try:
                client = _build_client(name, cfg)
                await client.connect()
                self._clients[name] = client
                logger.info("Connected to MCP server: %s (%d tools)", name, len(client.get_tools()))
            except Exception as exc:
                logger.error("Failed to connect to MCP server '%s': %s", name, exc)

    def get_tools(self, server_name: str) -> list:
        """Return tools for a specific MCP server."""
        client = self._clients.get(server_name)
        return client.get_tools() if client else []

    def register_all_into(self, tool_registry: ToolRegistry) -> None:
        """Register all MCP tools into the given ToolRegistry."""
        count = 0
        for name, client in self._clients.items():
            for tool in client.get_tools():
                tool_registry.register(tool)
                count += 1
        logger.info("Registered %d MCP tools into ToolRegistry", count)

    def list_servers(self) -> list[str]:
        """Names of all configured MCP servers."""
        return list(self._config.keys())

    def is_connected(self, server_name: str) -> bool:
        return server_name in self._clients

    async def close_all(self) -> None:
        """Close all MCP server connections."""
        for name, client in list(self._clients.items()):
            try:
                await client.close()
                logger.debug("Closed MCP connection: %s", name)
            except Exception as exc:
                logger.warning("Error closing MCP connection '%s': %s", name, exc)
        self._clients.clear()


def _build_client(name: str, cfg: McpServerConfig) -> McpClient:
    """Build an McpClient from server config."""
    if cfg.transport == "stdio":
        if not cfg.command:
            raise ValueError(f"MCP server '{name}' with stdio transport requires 'command'")
        transport = StdioTransport(
            command=cfg.command,
            args=cfg.args,
            env=cfg.env or None,
        )
    elif cfg.transport == "http":
        if not cfg.url:
            raise ValueError(f"MCP server '{name}' with http transport requires 'url'")
        transport = HttpTransport(url=cfg.url)
    else:
        raise ValueError(f"Unknown MCP transport: {cfg.transport!r}")
    return McpClient(transport)
