"""SQLite helpers for the shopping skill.

Single shared database at ``$EYETOR_RUNTIME_DIR/shopping.db`` (defaults to
``~/.eyetor/shopping.db``). All schema is applied idempotently on connect.

This module is private (leading underscore) so it is not exposed as a public
script by the skill executor.
"""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


def runtime_dir() -> Path:
    raw = os.environ.get("EYETOR_RUNTIME_DIR")
    return Path(raw).expanduser() if raw else Path.home() / ".eyetor"


def db_path() -> Path:
    return runtime_dir() / "shopping.db"


_DDL = """
CREATE TABLE IF NOT EXISTS stores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    name_norm   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stores_norm ON stores(name_norm);

CREATE TABLE IF NOT EXISTS products (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT NOT NULL UNIQUE,
    canonical_norm  TEXT NOT NULL,
    category        TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_products_norm ON products(canonical_norm);

CREATE TABLE IF NOT EXISTS product_aliases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  INTEGER NOT NULL,
    alias       TEXT NOT NULL,
    alias_norm  TEXT NOT NULL UNIQUE,
    source      TEXT NOT NULL DEFAULT 'manual',
    created_at  TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_aliases_product ON product_aliases(product_id);

CREATE TABLE IF NOT EXISTS receipts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id      INTEGER NOT NULL,
    purchased_at  TEXT NOT NULL,
    total         REAL,
    image_path    TEXT,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (store_id) REFERENCES stores(id)
);
CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(purchased_at);
CREATE INDEX IF NOT EXISTS idx_receipts_store ON receipts(store_id);

CREATE TABLE IF NOT EXISTS purchases (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id     INTEGER NOT NULL,
    product_id     INTEGER,
    raw_name       TEXT NOT NULL,
    raw_name_norm  TEXT NOT NULL,
    price          REAL NOT NULL,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (receipt_id) REFERENCES receipts(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_purch_receipt ON purchases(receipt_id);
CREATE INDEX IF NOT EXISTS idx_purch_product ON purchases(product_id);
CREATE INDEX IF NOT EXISTS idx_purch_raw_norm ON purchases(raw_name_norm);

CREATE TABLE IF NOT EXISTS shopping_list (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    text_norm   TEXT NOT NULL,
    product_id  INTEGER,
    quantity    INTEGER NOT NULL DEFAULT 1,
    added_at    TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_list_norm ON shopping_list(text_norm);
CREATE INDEX IF NOT EXISTS idx_list_product ON shopping_list(product_id);
"""


# Fuzzy match cutoffs. Tuned for short product names.
AUTO_LINK_THRESHOLD = 0.85
SUGGEST_THRESHOLD = 0.70


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)
    conn.commit()
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# Count units that mean "how many bought" (stripped before matching).
# Size/volume units (L, KG, G, ML, CL) are *preserved* because
# "Leche 1L" and "Leche 2L" are different products.
_COUNT_UNIT_RE = re.compile(
    r"\b\d+\s*(UDS|UD|UNIDADES|UNIDAD|UNID|UNI|U|X)\b",
    re.IGNORECASE,
)


def normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.upper()
    s = re.sub(r"[^\w\s]", " ", s)
    s = _COUNT_UNIT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_match(
    query_norm: str,
    candidates: list[tuple[int, str]],
    threshold: float = SUGGEST_THRESHOLD,
) -> list[tuple[int, str, float]]:
    """Return [(id, candidate_norm, score)] descending, filtered by threshold."""
    if not query_norm:
        return []
    out: list[tuple[int, str, float]] = []
    matcher = SequenceMatcher(None, query_norm, "")
    matcher.set_seq1(query_norm)
    for cid, cnorm in candidates:
        matcher.set_seq2(cnorm)
        score = matcher.ratio()
        if score >= threshold:
            out.append((cid, cnorm, score))
    out.sort(key=lambda r: r[2], reverse=True)
    return out


def get_or_create_store(conn: sqlite3.Connection, name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("store name is required")
    norm = normalize(name)
    row = conn.execute("SELECT id FROM stores WHERE name_norm = ?", (norm,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO stores (name, name_norm, created_at) VALUES (?, ?, ?)",
        (name, norm, now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def find_product_by_name(conn: sqlite3.Connection, name_norm: str) -> int | None:
    """Exact match against canonical_norm, then alias_norm. None if no match."""
    if not name_norm:
        return None
    row = conn.execute(
        "SELECT id FROM products WHERE canonical_norm = ?", (name_norm,)
    ).fetchone()
    if row:
        return int(row["id"])
    row = conn.execute(
        "SELECT product_id FROM product_aliases WHERE alias_norm = ?", (name_norm,)
    ).fetchone()
    if row:
        return int(row["product_id"])
    return None


def suggest_canonical(
    conn: sqlite3.Connection,
    name_norm: str,
    threshold: float = AUTO_LINK_THRESHOLD,
) -> dict | None:
    """Best fuzzy match against canonicals + aliases. None if below threshold."""
    if not name_norm:
        return None
    cands: list[tuple[int, str]] = []
    for r in conn.execute("SELECT id, canonical_norm FROM products"):
        cands.append((int(r["id"]), r["canonical_norm"]))
    for r in conn.execute("SELECT product_id, alias_norm FROM product_aliases"):
        cands.append((int(r["product_id"]), r["alias_norm"]))
    matches = fuzzy_match(name_norm, cands, threshold=threshold)
    if not matches:
        return None
    pid, _, score = matches[0]
    row = conn.execute(
        "SELECT id, canonical_name FROM products WHERE id = ?", (pid,)
    ).fetchone()
    if not row:
        return None
    return {"id": int(row["id"]), "name": row["canonical_name"], "score": score}


def get_or_create_product(
    conn: sqlite3.Connection, canonical_name: str, category: str | None = None
) -> int:
    canonical_name = canonical_name.strip()
    if not canonical_name:
        raise ValueError("canonical_name is required")
    norm = normalize(canonical_name)
    row = conn.execute(
        "SELECT id FROM products WHERE canonical_norm = ?", (norm,)
    ).fetchone()
    if row:
        return int(row["id"])
    ts = now_iso()
    cur = conn.execute(
        "INSERT INTO products (canonical_name, canonical_norm, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (canonical_name, norm, category, ts, ts),
    )
    conn.commit()
    return int(cur.lastrowid)


def add_alias(
    conn: sqlite3.Connection,
    product_id: int,
    alias: str,
    source: str = "manual",
) -> int | None:
    """Insert alias. Returns id or None if alias_norm already exists."""
    alias = alias.strip()
    if not alias:
        raise ValueError("alias is required")
    alias_norm = normalize(alias)
    existing = conn.execute(
        "SELECT id, product_id FROM product_aliases WHERE alias_norm = ?", (alias_norm,)
    ).fetchone()
    if existing:
        return None
    cur = conn.execute(
        "INSERT INTO product_aliases (product_id, alias, alias_norm, source, created_at) VALUES (?, ?, ?, ?, ?)",
        (product_id, alias, alias_norm, source, now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def merge_products(conn: sqlite3.Connection, from_id: int, into_id: int) -> dict:
    """Reassign all FKs from `from_id` to `into_id`, then delete `from_id`."""
    if from_id == into_id:
        raise ValueError("from-id and into-id must differ")
    src = conn.execute("SELECT id FROM products WHERE id = ?", (from_id,)).fetchone()
    dst = conn.execute("SELECT id FROM products WHERE id = ?", (into_id,)).fetchone()
    if not src or not dst:
        raise ValueError("product not found")
    purch = conn.execute(
        "UPDATE purchases SET product_id = ? WHERE product_id = ?",
        (into_id, from_id),
    ).rowcount
    listed = conn.execute(
        "UPDATE shopping_list SET product_id = ? WHERE product_id = ?",
        (into_id, from_id),
    ).rowcount
    # Aliases: re-point, but UNIQUE(alias_norm) may collide. Drop dupes.
    aliases_from = conn.execute(
        "SELECT id, alias_norm FROM product_aliases WHERE product_id = ?", (from_id,)
    ).fetchall()
    moved = 0
    dropped = 0
    for a in aliases_from:
        dup = conn.execute(
            "SELECT id FROM product_aliases WHERE alias_norm = ? AND product_id = ?",
            (a["alias_norm"], into_id),
        ).fetchone()
        if dup:
            conn.execute("DELETE FROM product_aliases WHERE id = ?", (a["id"],))
            dropped += 1
        else:
            conn.execute(
                "UPDATE product_aliases SET product_id = ? WHERE id = ?",
                (into_id, a["id"]),
            )
            moved += 1
    conn.execute("DELETE FROM products WHERE id = ?", (from_id,))
    conn.commit()
    return {
        "purchases_moved": purch,
        "list_items_moved": listed,
        "aliases_moved": moved,
        "aliases_dropped": dropped,
    }


def insert_receipt(
    conn: sqlite3.Connection,
    *,
    store_id: int,
    purchased_at: str,
    total: float | None,
    image_path: str | None,
) -> int:
    cur = conn.execute(
        "INSERT INTO receipts (store_id, purchased_at, total, image_path, created_at) VALUES (?, ?, ?, ?, ?)",
        (store_id, purchased_at, total, image_path, now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_purchase(
    conn: sqlite3.Connection,
    *,
    receipt_id: int,
    product_id: int | None,
    raw_name: str,
    price: float,
) -> int:
    cur = conn.execute(
        "INSERT INTO purchases (receipt_id, product_id, raw_name, raw_name_norm, price, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (receipt_id, product_id, raw_name, normalize(raw_name), price, now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)
