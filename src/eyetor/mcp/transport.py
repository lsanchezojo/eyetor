"""MCP transport layer: stdio and HTTP."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class BaseTransport(ABC):
    """Abstract MCP transport."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize the transport connection."""

    @abstractmethod
    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message."""

    @abstractmethod
    async def receive(self) -> dict[str, Any]:
        """Receive the next JSON-RPC message."""

    @abstractmethod
    async def close(self) -> None:
        """Close the transport."""


class StdioTransport(BaseTransport):
    """MCP transport over subprocess stdin/stdout (JSON-RPC per line)."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._args = args or []
        self._env = env
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        import os
        process_env = dict(os.environ)
        if self._env:
            process_env.update(self._env)
        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_env,
        )
        logger.debug("Started MCP stdio process: %s %s", self._command, " ".join(self._args))

    async def send(self, message: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("Transport not started")
        line = json.dumps(message) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        if not self._proc or not self._proc.stdout:
            raise RuntimeError("Transport not started")
        line = await self._proc.stdout.readline()
        if not line:
            raise EOFError("MCP server process closed stdout")
        return json.loads(line.decode())

    async def close(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._proc.kill()
            self._proc = None


class HttpTransport(BaseTransport):
    """MCP transport over HTTP POST (Streamable HTTP transport)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: httpx.AsyncClient | None = None
        self._pending_responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.debug("MCP HTTP transport ready: %s", self._url)

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and return the response."""
        if not self._client:
            raise RuntimeError("Transport not started")
        response = await self._client.post(
            self._url,
            json=message,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()

    async def receive(self) -> dict[str, Any]:
        """Not used for HTTP transport (request/response model)."""
        return await self._pending_responses.get()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
