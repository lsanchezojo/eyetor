"""MCP HTTP transport: auth headers, SSE responses, session id; env defaults."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from eyetor.config import _resolve_env_vars
from eyetor.mcp.client import McpClient
from eyetor.mcp.transport import HttpTransport, _parse_sse_response


def _run(coro):
    return asyncio.run(coro)


def test_parse_sse_picks_matching_id():
    body = (
        'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/x"}\n\n'
        'event: message\ndata: {"jsonrpc":"2.0","id":7,"result":{"ok":true}}\n\n'
    )
    assert _parse_sse_response(body, 7)["result"] == {"ok": True}


def test_parse_sse_without_match_raises():
    with pytest.raises(RuntimeError):
        _parse_sse_response('data: {"jsonrpc":"2.0","id":1}\n\n', 2)


def test_env_default(monkeypatch):
    monkeypatch.delenv("EYETOR_TEST_URL", raising=False)
    assert _resolve_env_vars("${EYETOR_TEST_URL:-http://a:1}/x") == "http://a:1/x"
    monkeypatch.setenv("EYETOR_TEST_URL", "http://b:2")
    assert _resolve_env_vars("${EYETOR_TEST_URL:-http://a:1}/x") == "http://b:2/x"
    assert _resolve_env_vars("${EYETOR_TEST_URL}") == "http://b:2"


def test_client_handshake_headers_and_session():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        msg = json.loads(request.content)
        if "id" not in msg:
            return httpx.Response(202)
        if msg["method"] == "initialize":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": msg["id"], "result": {}},
                headers={"Mcp-Session-Id": "abc"},
            )
        tools = {"tools": [{"name": "HassTurnOn", "inputSchema": {"type": "object"}}]}
        return httpx.Response(
            200,
            text=f'data: {{"jsonrpc":"2.0","id":{msg["id"]},"result":{json.dumps(tools)}}}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

    transport = HttpTransport("http://ha/api/mcp", headers={"Authorization": "Bearer t"})

    async def go():
        await transport.start()
        transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = McpClient(transport)
        await client._initialize()
        await client._discover_tools()
        return client

    client = _run(go())
    assert [t.name for t in client.get_tools()] == ["HassTurnOn"]
    assert all(r.headers["Authorization"] == "Bearer t" for r in seen)
    assert "Mcp-Session-Id" not in seen[0].headers
    assert seen[1].headers["Mcp-Session-Id"] == "abc"  # notifications/initialized
    assert seen[2].headers["Mcp-Session-Id"] == "abc"
