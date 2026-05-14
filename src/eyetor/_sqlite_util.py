"""Shared helpers for SQLite stores."""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


def apply_concurrency_pragmas(conn: sqlite3.Connection) -> None:
    """Enable WAL + sane defaults so multiple processes can share the DB.

    journal_mode=WAL is persisted in the file header (one-time effect); the
    other two are per-connection and must be set every time.
    """
    try:
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if mode and mode[0].lower() != "wal":
            logger.warning(
                "Could not enable WAL on SQLite (got %s); concurrency may suffer.",
                mode[0],
            )
    except sqlite3.DatabaseError as exc:
        logger.warning("PRAGMA journal_mode=WAL failed: %s", exc)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
