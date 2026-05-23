#!/usr/bin/env python3
"""Receipt ingestion.

Subcommands:
    receipt.py add --store NAME [--date YYYY-MM-DD] --items '[...]'
                   [--total FLOAT] [--image-path /abs/path]
    receipt.py reconfirm --image-path /abs/path
    receipt.py undo --receipt-id N

`add` validates the payload: the date must be known before insertion, every
item must have a numeric price, and if `--total` is given it must agree with
the sum of item prices within 0.05. On validation failure the script returns
``needs_reconfirm`` and inserts nothing. The caller (the agent) decides whether
to call `reconfirm` to re-read the image with the vision model.

Each unit purchased becomes one row in `purchases`. If the user bought three
identical items, the `--items` list must contain three entries (the agent is
responsible for expanding "2 x Leche" into two entries).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _db  # noqa: E402


PRICE_TOLERANCE = 0.05  # euros; sum(items) vs total
RECONCILE_THRESHOLD = 0.80
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _ok(**payload) -> None:
    print(json.dumps({"ok": True, **payload}, ensure_ascii=False))


def _err(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))


def _validate_image_path(raw: str | None) -> str | None:
    """Reject paths outside ``~/.eyetor/images/`` to limit blast radius."""
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    images_dir = (_db.runtime_dir() / "images").resolve()
    try:
        p.relative_to(images_dir)
    except ValueError:
        raise ValueError(f"image path must be under {images_dir}")
    if not p.is_file():
        raise ValueError(f"image not found: {p}")
    return str(p)


def _parse_items(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--items is not valid JSON: {e}")
    if not isinstance(data, list) or not data:
        raise ValueError("--items must be a non-empty JSON array")
    out = []
    for i, it in enumerate(data):
        if not isinstance(it, dict):
            raise ValueError(f"items[{i}] is not an object")
        name = (it.get("name") or "").strip()
        if not name:
            raise ValueError(f"items[{i}].name is empty")
        out.append({"name": name, "price": it.get("price")})
    return out


def _items_need_reconfirm(items: list[dict], total: float | None) -> str | None:
    """Return reason if reconfirm is needed, else None."""
    missing = [i for i, it in enumerate(items) if it.get("price") is None]
    if missing:
        return f"missing price for {len(missing)} item(s)"
    try:
        prices = [float(it["price"]) for it in items]
    except (TypeError, ValueError):
        return "non-numeric price in items"
    if total is not None:
        diff = abs(sum(prices) - float(total))
        if diff > PRICE_TOLERANCE:
            return f"total mismatch: sum={sum(prices):.2f} given={float(total):.2f}"
    return None


def _reconcile_score(list_norm: str, raw_norm: str) -> float:
    """Score how well a (usually short) list item matches a purchase name.

    Uses containment: longest common substring length / len(list_norm). This
    handles "PAN" matching "PAN BLANCO 500G" (containment 1.0) while still
    requiring real overlap for the inverse case. Falls back to plain
    SequenceMatcher.ratio if list_norm is empty (shouldn't happen).
    """
    if not list_norm or not raw_norm:
        return 0.0
    from difflib import SequenceMatcher
    m = SequenceMatcher(None, list_norm, raw_norm)
    block = m.find_longest_match(0, len(list_norm), 0, len(raw_norm))
    containment = block.size / len(list_norm)
    # Also consider the symmetric ratio so "leche entera" still wins over noise.
    ratio = m.ratio()
    return max(containment, ratio)


def _reconcile_with_list(conn, inserted_rows: list[dict]) -> list[dict]:
    """For each shopping-list item, find at most one matching purchase row.

    Match strategy (in order):
      1. Both list and purchase have the same product_id.
      2. Containment / fuzzy match between text_norm and raw_name_norm
         above RECONCILE_THRESHOLD.

    Each list item is reported at most once.
    """
    list_rows = conn.execute(
        "SELECT id, text, text_norm, product_id FROM shopping_list"
    ).fetchall()
    if not list_rows:
        return []

    out: list[dict] = []
    used_purchase_ids: set[int] = set()
    for lr in list_rows:
        best: tuple[dict, float] | None = None
        for p in inserted_rows:
            if p["id"] in used_purchase_ids:
                continue
            if lr["product_id"] is not None and p["product_id"] == lr["product_id"]:
                best = (p, 1.0)
                break
            score = _reconcile_score(lr["text_norm"], p["raw_name_norm"])
            if score >= RECONCILE_THRESHOLD and (best is None or score > best[1]):
                best = (p, score)
        if best is not None:
            p, score = best
            used_purchase_ids.add(p["id"])
            out.append({
                "list_item_id": int(lr["id"]),
                "list_text": lr["text"],
                "matched_raw": p["raw_name"],
                "purchase_id": p["id"],
                "score": round(score, 3),
            })
    return out


def cmd_add(args: argparse.Namespace) -> None:
    store = (args.store or "").strip()
    date = (args.date or "").strip()
    if not store:
        _err("--store is required")
        return
    if not date:
        print(
            json.dumps(
                {"ok": True, "needs_reconfirm": True, "reason": "missing date"},
                ensure_ascii=False,
            )
        )
        return
    if not DATE_RE.match(date):
        _err("--date must be YYYY-MM-DD")
        return
    try:
        items = _parse_items(args.items or "")
    except ValueError as e:
        _err(str(e))
        return
    try:
        image_path = _validate_image_path(args.image_path)
    except ValueError as e:
        _err(str(e))
        return
    total = float(args.total) if args.total is not None else None

    reason = _items_need_reconfirm(items, total)
    if reason is not None:
        # Don't insert anything; the agent decides whether to call reconfirm.
        print(
            json.dumps(
                {"ok": True, "needs_reconfirm": True, "reason": reason},
                ensure_ascii=False,
            )
        )
        return

    conn = _db.connect()
    try:
        store_id = _db.get_or_create_store(conn, store)
        receipt_id = _db.insert_receipt(
            conn,
            store_id=store_id,
            purchased_at=date,
            total=total,
            image_path=image_path,
        )
        inserted: list[dict] = []
        for it in items:
            name = it["name"]
            name_norm = _db.normalize(name)
            product_id = _db.find_product_by_name(conn, name_norm)
            if product_id is None:
                sug = _db.suggest_canonical(conn, name_norm,
                                            threshold=_db.AUTO_LINK_THRESHOLD)
                if sug is not None:
                    product_id = int(sug["id"])
            purchase_id = _db.insert_purchase(
                conn,
                receipt_id=receipt_id,
                product_id=product_id,
                raw_name=name,
                price=float(it["price"]),
            )
            inserted.append({
                "id": purchase_id,
                "product_id": product_id,
                "raw_name": name,
                "raw_name_norm": name_norm,
                "price": float(it["price"]),
            })
        reconcile = _reconcile_with_list(conn, inserted)
        _ok(
            receipt_id=receipt_id,
            inserted=len(inserted),
            purchase_ids=[r["id"] for r in inserted],
            reconcile=reconcile,
        )
    finally:
        conn.close()


def cmd_reconfirm(args: argparse.Namespace) -> None:
    try:
        image_path = _validate_image_path(args.image_path)
    except ValueError as e:
        _err(str(e))
        return
    if image_path is None:
        _err("--image-path is required")
        return
    # Lazy import: _vision pulls config and urllib. Tests that don't exercise
    # reconfirm shouldn't pay for that cost or its potential errors.
    try:
        import _vision  # type: ignore
    except Exception as e:
        _err(f"_vision import failed: {e}")
        return
    try:
        result = _vision.extract_ticket_json(Path(image_path))
    except Exception as e:
        _err(f"vision call failed: {type(e).__name__}: {e}")
        return
    if "error" in result:
        _err(result["error"])
        return
    _ok(**result)


def cmd_undo(args: argparse.Namespace) -> None:
    rid = int(args.receipt_id)
    conn = _db.connect()
    try:
        row = conn.execute("SELECT id FROM receipts WHERE id = ?", (rid,)).fetchone()
        if not row:
            _err(f"receipt {rid} not found")
            return
        # CASCADE on purchases.receipt_id removes all rows.
        conn.execute("DELETE FROM receipts WHERE id = ?", (rid,))
        conn.commit()
        _ok(receipt_id=rid)
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Receipt ingestion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Register a receipt and its items")
    p_add.add_argument("--store", required=True)
    p_add.add_argument("--date", default="")
    p_add.add_argument("--items", required=True, help="JSON array of {name, price}")
    p_add.add_argument("--total", type=float, default=None)
    p_add.add_argument("--image-path", default=None)
    p_add.set_defaults(func=cmd_add)

    p_re = sub.add_parser("reconfirm", help="Re-extract a ticket from its image")
    p_re.add_argument("--image-path", required=True)
    p_re.set_defaults(func=cmd_reconfirm)

    p_undo = sub.add_parser("undo", help="Delete a previously stored receipt")
    p_undo.add_argument("--receipt-id", type=int, required=True)
    p_undo.set_defaults(func=cmd_undo)

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
