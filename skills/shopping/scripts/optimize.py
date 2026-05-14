#!/usr/bin/env python3
"""Plan the shopping list across stores.

Subcommands:
    optimize.py plan --mode cheapest
    optimize.py plan --mode fewer-stores [--max-stores N]

Both modes start by fetching the current shopping list and the latest
known price per store for each list item (resolving via `product_id` if
linked, otherwise containment match on `purchases.raw_name_norm`).

- `cheapest`: pick the store with the lowest known price per item.
- `fewer-stores`: greedy set-cover. At each step, pick the store that
  covers the most still-uncovered items; ties broken by total price.
  `--max-stores N` caps the number of stores; leftover items end up in
  `missing`.

Items without any matching purchase end up in `missing` for both modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _db  # noqa: E402
import price as _price  # noqa: E402  # reuse _fetch_matches + _aggregate_by_store


def _ok(**payload) -> None:
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False))


def _err(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))


def _list_items(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, text, text_norm, product_id, quantity FROM shopping_list "
        "ORDER BY added_at ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def _prices_per_store(conn, item: dict) -> list[dict]:
    """Latest known price per store for this list item."""
    rows = _price._fetch_matches(conn, item["text"], None, None)
    if not rows:
        return []
    return _price._aggregate_by_store(rows, "last-known")


def _plan_cheapest(items_with_prices: list[tuple[dict, list[dict]]]) -> dict:
    by_store: dict[str, dict] = {}
    missing: list[dict] = []
    total = 0.0
    for item, prices in items_with_prices:
        if not prices:
            missing.append({"list_item_id": item["id"], "text": item["text"]})
            continue
        best = prices[0]  # _aggregate_by_store returns sorted ascending
        qty = max(1, int(item["quantity"]))
        line_total = best["price"] * qty
        total += line_total
        entry = by_store.setdefault(best["store"], {"store": best["store"],
                                                    "items": [], "subtotal": 0.0})
        entry["items"].append({
            "list_item_id": item["id"],
            "text": item["text"],
            "quantity": qty,
            "price": best["price"],
            "line_total": round(line_total, 2),
            "seen_at": best["seen_at"],
            "raw_name": best["raw_name"],
        })
        entry["subtotal"] = round(entry["subtotal"] + line_total, 2)

    plan = sorted(by_store.values(), key=lambda r: -r["subtotal"])
    return {
        "mode": "cheapest",
        "plan": plan,
        "total": round(total, 2),
        "missing": missing,
    }


def _plan_fewer_stores(items_with_prices: list[tuple[dict, list[dict]]],
                       max_stores: int | None) -> dict:
    # store -> { item_id -> (price, raw_name, seen_at) }
    coverage: dict[str, dict[int, dict]] = {}
    item_meta: dict[int, dict] = {}
    for item, prices in items_with_prices:
        item_meta[item["id"]] = item
        for p in prices:
            coverage.setdefault(p["store"], {})[item["id"]] = {
                "price": p["price"],
                "raw_name": p["raw_name"],
                "seen_at": p["seen_at"],
            }

    uncovered = {item["id"] for item, prices in items_with_prices if prices}
    chosen_stores: list[str] = []
    used: dict[str, list[int]] = {}

    while uncovered:
        if max_stores is not None and len(chosen_stores) >= max_stores:
            break
        best_store: str | None = None
        best_gain = -1
        best_cost = 0.0
        for store, offered in coverage.items():
            if store in chosen_stores:
                continue
            covered_here = uncovered & set(offered.keys())
            if not covered_here:
                continue
            gain = len(covered_here)
            cost = sum(offered[iid]["price"] for iid in covered_here)
            if gain > best_gain or (gain == best_gain and cost < best_cost):
                best_store, best_gain, best_cost = store, gain, cost
        if best_store is None:
            break
        chosen_stores.append(best_store)
        used[best_store] = sorted(uncovered & set(coverage[best_store].keys()))
        uncovered -= set(used[best_store])

    plan = []
    total = 0.0
    for store in chosen_stores:
        items_block = []
        subtotal = 0.0
        for iid in used[store]:
            item = item_meta[iid]
            qty = max(1, int(item["quantity"]))
            offer = coverage[store][iid]
            line_total = offer["price"] * qty
            subtotal += line_total
            items_block.append({
                "list_item_id": iid,
                "text": item["text"],
                "quantity": qty,
                "price": offer["price"],
                "line_total": round(line_total, 2),
                "seen_at": offer["seen_at"],
                "raw_name": offer["raw_name"],
            })
        total += subtotal
        plan.append({"store": store, "items": items_block,
                     "subtotal": round(subtotal, 2)})

    missing: list[dict] = []
    for item, prices in items_with_prices:
        if not prices or item["id"] in uncovered:
            missing.append({"list_item_id": item["id"], "text": item["text"]})
    return {
        "mode": "fewer-stores",
        "plan": plan,
        "total": round(total, 2),
        "missing": missing,
        "max_stores": max_stores,
    }


def cmd_plan(args: argparse.Namespace) -> None:
    mode = args.mode
    if mode not in ("cheapest", "fewer-stores"):
        _err(f"invalid --mode: {mode}")
        return
    conn = _db.connect()
    try:
        items = _list_items(conn)
        if not items:
            _ok(mode=mode, plan=[], total=0.0, missing=[])
            return
        items_with_prices = [(it, _prices_per_store(conn, it)) for it in items]
        if mode == "cheapest":
            result = _plan_cheapest(items_with_prices)
        else:
            result = _plan_fewer_stores(items_with_prices, args.max_stores)
        _ok(**result)
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Shopping plan optimizer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="Compute a plan for the current list")
    p.add_argument("--mode", required=True, choices=["cheapest", "fewer-stores"])
    p.add_argument("--max-stores", type=int, default=None)
    p.set_defaults(func=cmd_plan)

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
