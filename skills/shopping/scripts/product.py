#!/usr/bin/env python3
"""Catalog management: canonical products and aliases.

Subcommands:
    product.py canonical create --name "Leche entera 1L" [--category lacteos]
    product.py canonical list [--search "leche"]
    product.py canonical merge --from-id N --into-id M
    product.py alias --alias "le ent 1l" --canonical-id N
    product.py alias list --canonical-id N
    product.py alias delete --id N

`merge` reassigns purchases.product_id and shopping_list.product_id from the
source canonical to the destination, then deletes the source. Alias UNIQUE
collisions are dropped silently (the destination already had that alias).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _db  # noqa: E402


def _ok(**payload) -> None:
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False))


def _err(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))


# --- canonical -------------------------------------------------------------


def cmd_canonical_create(args: argparse.Namespace) -> None:
    name = (args.name or "").strip()
    if not name:
        _err("--name is required")
        return
    conn = _db.connect()
    try:
        pid = _db.get_or_create_product(conn, name, args.category)
        row = conn.execute(
            "SELECT id, canonical_name, category FROM products WHERE id = ?", (pid,)
        ).fetchone()
        _ok(id=int(row["id"]), name=row["canonical_name"], category=row["category"])
    finally:
        conn.close()


def cmd_canonical_list(args: argparse.Namespace) -> None:
    conn = _db.connect()
    try:
        if args.search:
            q = f"%{_db.normalize(args.search)}%"
            rows = conn.execute(
                "SELECT id, canonical_name, category FROM products "
                "WHERE canonical_norm LIKE ? ORDER BY canonical_name ASC",
                (q,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, canonical_name, category FROM products "
                "ORDER BY canonical_name ASC"
            ).fetchall()
        items = [dict(r) for r in rows]
        _ok(items=items, count=len(items))
    finally:
        conn.close()


def cmd_canonical_merge(args: argparse.Namespace) -> None:
    if args.from_id is None or args.into_id is None:
        _err("--from-id and --into-id are required")
        return
    conn = _db.connect()
    try:
        result = _db.merge_products(conn, int(args.from_id), int(args.into_id))
        _ok(**result)
    finally:
        conn.close()


# --- aliases ---------------------------------------------------------------


def cmd_alias_add(args: argparse.Namespace) -> None:
    alias = (args.alias or "").strip()
    cid = args.canonical_id
    if not alias or cid is None:
        _err("--alias and --canonical-id are required")
        return
    conn = _db.connect()
    try:
        row = conn.execute("SELECT id FROM products WHERE id = ?", (int(cid),)).fetchone()
        if not row:
            _err(f"canonical_id {cid} does not exist")
            return
        new_id = _db.add_alias(conn, int(cid), alias)
        if new_id is None:
            _err(f"alias '{alias}' already exists (UNIQUE constraint)")
            return
        _ok(id=new_id, alias=alias, canonical_id=int(cid))
    finally:
        conn.close()


def cmd_alias_list(args: argparse.Namespace) -> None:
    if args.canonical_id is None:
        _err("--canonical-id is required")
        return
    conn = _db.connect()
    try:
        rows = conn.execute(
            "SELECT id, alias, alias_norm, source, created_at FROM product_aliases "
            "WHERE product_id = ? ORDER BY created_at ASC",
            (int(args.canonical_id),),
        ).fetchall()
        items = [dict(r) for r in rows]
        _ok(items=items, count=len(items))
    finally:
        conn.close()


def cmd_alias_delete(args: argparse.Namespace) -> None:
    if args.id is None:
        _err("--id is required")
        return
    conn = _db.connect()
    try:
        cur = conn.execute("DELETE FROM product_aliases WHERE id = ?", (int(args.id),))
        conn.commit()
        if cur.rowcount == 0:
            _err(f"alias id {args.id} not found")
            return
        _ok(id=int(args.id))
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Catalog (canonical + alias)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # canonical group
    p_c = sub.add_parser("canonical", help="Canonical-product operations")
    sub_c = p_c.add_subparsers(dest="op", required=True)

    p_cc = sub_c.add_parser("create", help="Create or upsert a canonical product")
    p_cc.add_argument("--name", required=True)
    p_cc.add_argument("--category", default=None)
    p_cc.set_defaults(func=cmd_canonical_create)

    p_cl = sub_c.add_parser("list", help="List canonical products")
    p_cl.add_argument("--search", default=None)
    p_cl.set_defaults(func=cmd_canonical_list)

    p_cm = sub_c.add_parser("merge", help="Merge two canonicals")
    p_cm.add_argument("--from-id", type=int, required=True)
    p_cm.add_argument("--into-id", type=int, required=True)
    p_cm.set_defaults(func=cmd_canonical_merge)

    p_a = sub.add_parser("alias", help="Alias operations")
    sub_a = p_a.add_subparsers(dest="op", required=True)

    p_aa = sub_a.add_parser("add", help="Attach an alias to a canonical")
    p_aa.add_argument("--alias", required=True)
    p_aa.add_argument("--canonical-id", type=int, required=True)
    p_aa.set_defaults(func=cmd_alias_add)

    p_al = sub_a.add_parser("list", help="List aliases for a canonical")
    p_al.add_argument("--canonical-id", type=int, required=True)
    p_al.set_defaults(func=cmd_alias_list)

    p_ad = sub_a.add_parser("delete", help="Delete an alias by id")
    p_ad.add_argument("--id", type=int, required=True)
    p_ad.set_defaults(func=cmd_alias_delete)

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        if e.code not in (0, None):
            _err("invalid arguments")
        return int(e.code) if isinstance(e.code, int) else 2

    try:
        args.func(args)
    except Exception as e:  # noqa: BLE001
        _err(f"{type(e).__name__}: {e}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
