#!/usr/bin/env python3
"""Shopping list CRUD.

Subcommands:
    list.py add --text "leche entera 1L" [--canonical-id N] [--quantity N]
    list.py remove --ids 3,7,12
    list.py show [--with-canonical]
    list.py clear
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _db  # noqa: E402


def _ok(**payload) -> None:
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False))


def _err(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))


def cmd_add(args: argparse.Namespace) -> None:
    text = (args.text or "").strip()
    if not text:
        _err("--text is required and must be non-empty")
        return
    quantity = max(1, int(args.quantity or 1))
    text_norm = _db.normalize(text)
    conn = _db.connect()
    try:
        product_id: int | None = None
        auto_linked = False
        suggest = None
        if args.canonical_id:
            row = conn.execute(
                "SELECT id FROM products WHERE id = ?", (int(args.canonical_id),)
            ).fetchone()
            if not row:
                _err(f"canonical_id {args.canonical_id} does not exist")
                return
            product_id = int(row["id"])
        else:
            product_id = _db.find_product_by_name(conn, text_norm)
            if product_id is not None:
                auto_linked = True
            else:
                sug = _db.suggest_canonical(conn, text_norm, threshold=_db.AUTO_LINK_THRESHOLD)
                if sug is not None:
                    product_id = int(sug["id"])
                    auto_linked = True
                else:
                    sug = _db.suggest_canonical(conn, text_norm, threshold=_db.SUGGEST_THRESHOLD)
                    if sug is not None:
                        suggest = sug

        cur = conn.execute(
            "INSERT INTO shopping_list (text, text_norm, product_id, quantity, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (text, text_norm, product_id, quantity, _db.now_iso()),
        )
        conn.commit()
        new_id = int(cur.lastrowid)

        payload: dict = {
            "id": new_id,
            "text": text,
            "quantity": quantity,
            "product_id": product_id,
            "auto_linked": auto_linked,
        }
        if suggest is not None:
            payload["suggest_canonical"] = suggest
        _ok(**payload)
    finally:
        conn.close()


def _parse_id_list(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out


def cmd_remove(args: argparse.Namespace) -> None:
    ids = _parse_id_list(args.ids or "")
    if not ids:
        _err("--ids must contain at least one integer")
        return
    conn = _db.connect()
    try:
        placeholders = ",".join("?" * len(ids))
        cur = conn.execute(
            f"DELETE FROM shopping_list WHERE id IN ({placeholders})", ids
        )
        conn.commit()
        _ok(removed=cur.rowcount, ids=ids)
    finally:
        conn.close()


def cmd_show(args: argparse.Namespace) -> None:
    conn = _db.connect()
    try:
        if args.with_canonical:
            rows = conn.execute(
                "SELECT sl.id, sl.text, sl.quantity, sl.product_id, sl.added_at, "
                "       p.canonical_name "
                "FROM shopping_list sl "
                "LEFT JOIN products p ON p.id = sl.product_id "
                "ORDER BY sl.added_at ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, text, quantity, product_id, added_at FROM shopping_list "
                "ORDER BY added_at ASC"
            ).fetchall()
        items = [dict(r) for r in rows]
        _ok(items=items, count=len(items))
    finally:
        conn.close()


def cmd_clear(args: argparse.Namespace) -> None:
    conn = _db.connect()
    try:
        cur = conn.execute("DELETE FROM shopping_list")
        conn.commit()
        _ok(removed=cur.rowcount)
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Shopping list CRUD")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add an item to the list")
    p_add.add_argument("--text", required=True)
    p_add.add_argument("--canonical-id", type=int, default=None)
    p_add.add_argument("--quantity", type=int, default=1)
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("remove", help="Remove items by id (comma-separated)")
    p_rm.add_argument("--ids", required=True)
    p_rm.set_defaults(func=cmd_remove)

    p_show = sub.add_parser("show", help="Show all items in the list")
    p_show.add_argument("--with-canonical", action="store_true")
    p_show.set_defaults(func=cmd_show)

    p_clear = sub.add_parser("clear", help="Remove all items")
    p_clear.set_defaults(func=cmd_clear)

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse exits with code 2 on error; convert into our JSON error.
        if e.code not in (0, None):
            _err("invalid arguments")
        return int(e.code) if isinstance(e.code, int) else 2

    try:
        args.func(args)
    except Exception as e:  # noqa: BLE001 - last-resort safety net
        _err(f"{type(e).__name__}: {e}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
