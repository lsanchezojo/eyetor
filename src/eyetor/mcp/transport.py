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
    """MCP transport over HTTP POST (Streamable HTTP transport).

    Servers may answer with plain JSON or with a ``text/event-stream`` body
    carrying the JSON-RPC response as an SSE ``data:`` event. The
    ``Mcp-Session-Id`` header returned on ``initialize`` is echoed back on
    every later request.
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers = headers or {}
        self._session_id: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._pending_responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def start(self) -> None:
        self._session_id = None
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.debug("MCP HTTP transport ready: %s", self._url)

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self._headers,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _post(self, message: dict[str, Any]) -> httpx.Response:
        if not self._client:
            raise RuntimeError("Transport not started")
        try:
            response = await self._client.post(
                self._url, json=message, headers=self._request_headers()
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # RuntimeError so McpClient.call_tool retries with a fresh session
            raise RuntimeError(f"MCP HTTP error: {exc}") from exc
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        return response

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and return the response."""
        response = await self._post(message)
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            return _parse_sse_response(response.text, message.get("id"))
        return response.json()

    async def notify(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (server answers 202 with no body)."""
        await self._post(message)

    async def receive(self) -> dict[str, Any]:
        """Not used for HTTP transport (request/response model)."""
        return await self._pending_responses.get()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


def _parse_sse_response(body: str, request_id: Any) -> dict[str, Any]:
    """Return the JSON-RPC response matching ``request_id`` from an SSE body."""
    for event in body.split("\n\n"):
        data = "\n".join(
            line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")
        )
        if not data:
            continue
        payload = json.loads(data)
        if isinstance(payload, dict) and payload.get("id") == request_id:
            return payload
    raise RuntimeError("MCP SSE response without a matching JSON-RPC message")
