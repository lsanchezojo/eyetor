#!/usr/bin/env python3
"""Price history and cheapest-store queries.

Subcommands:
    price.py history --product "leche entera" [--store NAME] [--days N]
    price.py cheapest --product "leche entera"
                      [--strategy last-known|min-ever|min-last-30d]

Resolution order for `--product`:
  1. Exact match against `products.canonical_norm` or `product_aliases.alias_norm`
     → filter purchases by `product_id`.
  2. Fuzzy match (containment of normalized query in `purchases.raw_name_norm`)
     across raw names. Useful before the catalog is populated.

Output is always JSON-on-stdout, one line.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _db  # noqa: E402


MATCH_CONTAINMENT_THRESHOLD = 0.9  # how much of the query must appear in raw_name


def _ok(**payload) -> None:
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False))


def _err(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))


def _query_product_id(conn, query_norm: str) -> int | None:
    """Resolve a query string to a product_id via canonical/alias (exact only)."""
    pid = _db.find_product_by_name(conn, query_norm)
    if pid is not None:
        return pid
    return None


def _matches_purchase(query_norm: str, raw_norm: str) -> bool:
    """Containment-based match used when the product is not in the catalog."""
    if not query_norm or not raw_norm:
        return False
    if query_norm in raw_norm:
        return True
    from difflib import SequenceMatcher
    m = SequenceMatcher(None, query_norm, raw_norm)
    blk = m.find_longest_match(0, len(query_norm), 0, len(raw_norm))
    return (blk.size / len(query_norm)) >= MATCH_CONTAINMENT_THRESHOLD


def _fetch_matches(conn, query: str, store: str | None, since_iso: str | None) -> list[dict]:
    """Return matching purchase rows with store name and date."""
    query_norm = _db.normalize(query)
    if not query_norm:
        return []
    pid = _query_product_id(conn, query_norm)

    sql = (
        "SELECT p.id, p.raw_name, p.raw_name_norm, p.price, p.product_id, "
        "       r.purchased_at, s.name AS store "
        "FROM purchases p "
        "JOIN receipts r ON r.id = p.receipt_id "
        "JOIN stores s ON s.id = r.store_id "
        "WHERE 1=1 "
    )
    params: list = []
    if pid is not None:
        sql += "AND p.product_id = ? "
        params.append(pid)
    if store:
        sql += "AND s.name_norm = ? "
        params.append(_db.normalize(store))
    if since_iso:
        sql += "AND r.purchased_at >= ? "
        params.append(since_iso)
    sql += "ORDER BY r.purchased_at DESC, p.id DESC"

    rows = conn.execute(sql, params).fetchall()
    out: list[dict] = []
    for r in rows:
        if pid is None and not _matches_purchase(query_norm, r["raw_name_norm"]):
            continue
        out.append({
            "purchase_id": int(r["id"]),
            "raw_name": r["raw_name"],
            "price": float(r["price"]),
            "store": r["store"],
            "date": r["purchased_at"],
            "product_id": r["product_id"],
        })
    return out


def _since_iso(days: int | None) -> str | None:
    if not days or days <= 0:
        return None
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    return cutoff.isoformat()


def cmd_history(args: argparse.Namespace) -> None:
    query = (args.product or "").strip()
    if not query:
        _err("--product is required")
        return
    conn = _db.connect()
    try:
        rows = _fetch_matches(conn, query, args.store, _since_iso(args.days))
        _ok(matches=rows, count=len(rows))
    finally:
        conn.close()


def _aggregate_by_store(rows: list[dict], strategy: str) -> list[dict]:
    """Reduce purchase rows into one entry per store per strategy."""
    by_store: dict[str, list[dict]] = {}
    for r in rows:
        by_store.setdefault(r["store"], []).append(r)

    out: list[dict] = []
    for store, items in by_store.items():
        items_sorted = sorted(items, key=lambda r: (r["date"], r["purchase_id"]))
        if strategy == "last-known":
            chosen = items_sorted[-1]
            price = chosen["price"]
            seen_at = chosen["date"]
        elif strategy == "min-ever":
            chosen = min(items_sorted, key=lambda r: r["price"])
            price = chosen["price"]
            seen_at = chosen["date"]
        elif strategy == "min-last-30d":
            cutoff = _since_iso(30)
            recent = [r for r in items_sorted if r["date"] >= cutoff] if cutoff else items_sorted
            if not recent:
                continue
            chosen = min(recent, key=lambda r: r["price"])
            price = chosen["price"]
            seen_at = chosen["date"]
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        out.append({
            "store": store,
            "price": price,
            "seen_at": seen_at,
            "purchase_id": chosen["purchase_id"],
            "raw_name": chosen["raw_name"],
            "samples": len(items_sorted),
        })
    out.sort(key=lambda r: r["price"])
    return out


def cmd_cheapest(args: argparse.Namespace) -> None:
    query = (args.product or "").strip()
    if not query:
        _err("--product is required")
        return
    strategy = args.strategy or "last-known"
    if strategy not in ("last-known", "min-ever", "min-last-30d"):
        _err(f"invalid --strategy: {strategy}")
        return
    conn = _db.connect()
    try:
        rows = _fetch_matches(conn, query, None, None)
        if not rows:
            _ok(matches=[], cheapest=None, strategy=strategy)
            return
        aggregated = _aggregate_by_store(rows, strategy)
        cheapest = aggregated[0] if aggregated else None
        _ok(matches=aggregated, cheapest=cheapest, strategy=strategy)
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Price queries")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_h = sub.add_parser("history", help="List purchases matching a product")
    p_h.add_argument("--product", required=True)
    p_h.add_argument("--store", default=None)
    p_h.add_argument("--days", type=int, default=None)
    p_h.set_defaults(func=cmd_history)

    p_c = sub.add_parser("cheapest", help="Find the cheapest store for a product")
    p_c.add_argument("--product", required=True)
    p_c.add_argument("--strategy", default="last-known")
    p_c.set_defaults(func=cmd_cheapest)

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
