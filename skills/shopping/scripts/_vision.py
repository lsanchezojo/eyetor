"""Vision-LLM helper for receipt reconfirmation.

Reads `runtime.json` to discover the configured vision provider, encodes a
local image as base64, and asks the model for a strict JSON payload with
`store`, `date`, `total`, and `items`. The parser is split out so it can be
unit-tested without network access.

Stdlib-only: urllib + base64 + json.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


PROMPT = (
    "Analiza este ticket de compra y devuelve EXCLUSIVAMENTE un objeto JSON "
    "válido con esta forma:\n"
    "{\n"
    '  "store": "nombre de la tienda",\n'
    '  "date": "YYYY-MM-DD",\n'
    '  "total": 12.34,\n'
    '  "items": [\n'
    '    {"name": "nombre del producto", "price": 1.23}\n'
    "  ]\n"
    "}\n\n"
    "Reglas estrictas:\n"
    "- Devuelve SOLO el JSON, sin texto antes ni después, sin comentarios, "
    "sin bloques markdown.\n"
    "- Si un producto aparece N veces en el ticket, emite N entradas "
    "idénticas en \"items\" (una por unidad).\n"
    "- Cada item DEBE tener precio numérico (sin símbolo €). Si no puedes "
    "leer el precio de un item, OMÍTELO en lugar de inventarlo.\n"
    "- \"date\" debe ser YYYY-MM-DD. Convierte tú \"12/05/2026\" → "
    "\"2026-05-12\".\n"
    "- \"total\" es un número (sin €). Si no aparece en el ticket, omite la "
    "clave."
)


# --- runtime config --------------------------------------------------------


def _runtime_dir() -> Path:
    raw = os.environ.get("EYETOR_RUNTIME_DIR")
    return Path(raw).expanduser() if raw else Path.home() / ".eyetor"


def load_vision_config() -> dict:
    """Read runtime.json and return the vision provider block.

    Raises RuntimeError if the snapshot or `vision` block is missing.
    """
    path = _runtime_dir() / "runtime.json"
    if not path.exists():
        raise RuntimeError(f"runtime.json not found at {path}")
    try:
        snap = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"cannot read runtime.json: {e}")
    vision = snap.get("vision")
    if not vision or not vision.get("base_url"):
        raise RuntimeError("vision provider is not configured in runtime.json")
    return vision


# --- parsing (no network) --------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _find_json_object(text: str) -> str | None:
    """Return the first balanced {...} substring, or None if not found."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def parse_ticket_content(content: str) -> dict:
    """Parse the vision-LLM response into {store, date, total, items}.

    Returns ``{"error": "..."}`` on any failure (no exception).
    Does NOT validate prices/totals — that is the receipt.py job.
    """
    if not content or not content.strip():
        return {"error": "empty vision response"}
    text = _strip_fences(content)
    blob = _find_json_object(text)
    if blob is None:
        return {"error": "no JSON object in vision response"}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON: {e}"}
    if not isinstance(data, dict):
        return {"error": "JSON root is not an object"}

    store = (data.get("store") or "").strip() if isinstance(data.get("store"), str) else ""
    date = (data.get("date") or "").strip() if isinstance(data.get("date"), str) else ""
    items_raw = data.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        return {"error": "items missing or not a list"}

    items: list[dict] = []
    for it in items_raw:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        price = it.get("price")
        try:
            price_f = float(price) if price is not None else None
        except (TypeError, ValueError):
            price_f = None
        entry = {"name": name.strip()}
        if price_f is not None:
            entry["price"] = price_f
        items.append(entry)

    if not items:
        return {"error": "no usable items extracted"}

    total = data.get("total")
    try:
        total_f = float(total) if total is not None else None
    except (TypeError, ValueError):
        total_f = None

    out: dict = {"items": items}
    if store:
        out["store"] = store
    if date:
        out["date"] = date
    if total_f is not None:
        out["total"] = total_f
    return out


# --- network ---------------------------------------------------------------


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _post_chat(base_url: str, api_key: str, payload: dict, timeout: float = 120.0) -> str:
    """POST to {base_url}/chat/completions, return the assistant content."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("HTTP-Referer", "https://github.com/lsanchezojo/eyetor")
    req.add_header("X-Title", "Eyetor/shopping")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read()
    data = json.loads(raw.decode("utf-8"))
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected vision response shape: {e}")


def extract_ticket_json(image_path: Path) -> dict:
    """High-level: encode image, call vision LLM, parse, return dict.

    Returns ``{"error": "..."}`` on any failure path.
    """
    try:
        cfg = load_vision_config()
    except RuntimeError as e:
        return {"error": str(e)}
    try:
        img_b64 = _encode_image(image_path)
    except OSError as e:
        return {"error": f"cannot read image: {e}"}

    payload = {
        "model": cfg.get("model") or "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    try:
        content = _post_chat(cfg["base_url"], cfg.get("api_key") or "", payload)
    except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as e:
        return {"error": f"vision call failed: {type(e).__name__}: {e}"}
    return parse_ticket_content(content)
