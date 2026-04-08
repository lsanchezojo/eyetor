#!/usr/bin/env python3
"""Pre-tool-use hook: return cached result if available and not expired."""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def cache_key(tool_name: str, args_hash: str) -> str:
    return f"{tool_name}:{args_hash}"


def main() -> None:
    config = load_config()
    tool_name = os.environ.get("HOOK_TOOL_NAME", "")

    # Only cache configured tools
    cached_tools = config.get("cached_tools", [])
    if tool_name not in cached_tools:
        # Allow — not a cached tool
        print(json.dumps({"decision": "allow"}))
        return

    tool_input = os.environ.get("HOOK_TOOL_INPUT", "{}")
    args_hash = hashlib.sha256(tool_input.encode()).hexdigest()[:16]
    key = cache_key(tool_name, args_hash)

    db_path = Path(config.get("db_path", "~/.eyetor/cache.db")).expanduser()
    if not db_path.exists():
        print(json.dumps({"decision": "allow"}))
        return

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT result, created_at, ttl_seconds FROM tool_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()
    conn.close()

    if row is None:
        print(json.dumps({"decision": "allow"}))
        return

    result, created_at, ttl = row
    if time.time() - created_at > ttl:
        # Expired — let the tool run and post-hook will update the cache
        print(json.dumps({"decision": "allow"}))
        return

    # Cache hit — return cached result without executing the tool
    print(json.dumps({"decision": "provide_result", "result": result}))


if __name__ == "__main__":
    main()
