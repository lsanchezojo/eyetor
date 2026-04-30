"""Robust JSON extraction helpers for SLM-facing workflows."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in free-form text.

    Small local models often wrap JSON in Markdown or add prose before/after it.
    This helper first tries direct parsing, then fenced code blocks, then scans
    from each ``{`` using ``JSONDecoder.raw_decode`` so nested objects work.
    """
    if not text:
        return None
    stripped = text.strip()
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    for match in _FENCED_BLOCK_RE.finditer(stripped):
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            continue

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            data, _end = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None

