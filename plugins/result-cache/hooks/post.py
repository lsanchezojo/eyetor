#!/usr/bin/env python3
"""Post-tool-use hook: save tool result to cache."""

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

    cached_tools = config.get("cached_tools", [])
    if tool_name not in cached_tools:
        return

    tool_input = os.environ.get("HOOK_TOOL_INPUT", "{}")
    tool_result = os.environ.get("HOOK_TOOL_RESULT", "")

    # Skip caching if result is too large
    max_size = config.get("max_result_size_bytes", 1048576)
    if len(tool_result.encode()) > max_size:
        return

    # Skip caching error results
    try:
        parsed = json.loads(tool_result)
        if isinstance(parsed, dict) and "error" in parsed:
            return
    except (json.JSONDecodeError, ValueError):
        pass

    args_hash = hashlib.sha256(tool_input.encode()).hexdigest()[:16]
    key = cache_key(tool_name, args_hash)

    ttl_overrides = config.get("tool_ttl_overrides", {})
    ttl = ttl_overrides.get(tool_name, config.get("default_ttl_seconds", 3600))

    db_path = Path(config.get("db_path", "~/.eyetor/cache.db")).expanduser()
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT OR REPLACE INTO tool_cache (cache_key, tool_name, args_hash, result, created_at, ttl_seconds)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (key, tool_name, args_hash, tool_result, time.time(), ttl),
    )
    conn.commit()

    # Purge expired entries occasionally (~1% of writes)
    import random
    if random.random() < 0.01:
        conn.execute("DELETE FROM tool_cache WHERE created_at + ttl_seconds < ?", (time.time(),))
        conn.commit()

    conn.close()


if __name__ == "__main__":
    main()
