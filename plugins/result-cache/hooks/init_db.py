#!/usr/bin/env python3
"""Lifecycle init: create the cache SQLite database and table if needed."""

import json
import sqlite3
from pathlib import Path


def main() -> None:
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    db_path = Path(config.get("db_path", "~/.eyetor/cache.db")).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_cache (
            cache_key   TEXT PRIMARY KEY,
            tool_name   TEXT NOT NULL,
            args_hash   TEXT NOT NULL,
            result      TEXT NOT NULL,
            created_at  REAL NOT NULL,
            ttl_seconds INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_cache_tool ON tool_cache(tool_name)")
    conn.commit()
    conn.close()
    print("Cache DB initialized.")


if __name__ == "__main__":
    main()
