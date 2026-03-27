"""SSE and NDJSON stream parsers for LLM provider responses."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


async def parse_sse(response: httpx.Response) -> AsyncIterator[dict]:
    """Parse Server-Sent Events stream from an httpx streaming response.

    Yields parsed JSON data objects. Skips empty lines, comments,
    and [DONE] sentinel events.
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            # SSE comment — skip
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                logger.debug("Could not parse SSE data: %r", data)
                continue


async def parse_ndjson(response: httpx.Response) -> AsyncIterator[dict]:
    """Parse Newline-Delimited JSON stream (Ollama native format).

    Yields parsed JSON objects line by line.
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Could not parse NDJSON line: %r", line)
            continue


def extract_delta_content(chunk: dict) -> str:
    """Extract the text delta from an OpenAI-compatible SSE chunk."""
    try:
        choices = chunk.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].get("delta", {})
        return delta.get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def extract_delta_tool_calls(chunk: dict) -> list[dict]:
    """Extract tool call deltas from an OpenAI-compatible SSE chunk."""
    try:
        choices = chunk.get("choices", [])
        if not choices:
            return []
        delta = choices[0].get("delta", {})
        return delta.get("tool_calls") or []
    except (KeyError, IndexError, TypeError):
        return []
