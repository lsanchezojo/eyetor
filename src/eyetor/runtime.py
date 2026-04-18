"""Runtime config snapshot — exposes the resolved config to subprocesses.

Eyetor writes a JSON snapshot of the active configuration to a well-known
location on startup. Any subprocess (skill script, plugin hook, external
tool) can read it to discover provider URLs, model names, paths, or any
other runtime setting without hardcoding values or depending on env vars.

The snapshot path is `$EYETOR_RUNTIME_DIR/runtime.json`, defaulting to
`~/.eyetor/runtime.json`. Eyetor always exports `EYETOR_RUNTIME_DIR` into
its own process environment so child subprocesses inherit it.

Schema (stable keys, additive — never remove fields without a major bump):

    {
      "pid": int,
      "started_at": "ISO-8601 UTC",
      "default_provider": str,
      "providers": {
        "<name>": {"type": str, "base_url": str, "model": str, "api_key": str}
      },
      "vision": {"provider": str, "base_url": str, "model": str, "api_key": str} | null,
      "image": {"provider": str} | null,
      "paths": {"memory_db": str, "tracking_db": str, "sessions_dir": str},
      "knowledge": {"enabled": bool, "workspaces": [str, ...]}
    }

Callers should treat missing keys as optional and default gracefully.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUNTIME_FILENAME = "runtime.json"


def runtime_dir() -> Path:
    """Return the directory where the runtime snapshot lives."""
    override = os.environ.get("EYETOR_RUNTIME_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".eyetor"


def runtime_path() -> Path:
    """Return the absolute path to the runtime snapshot file."""
    return runtime_dir() / RUNTIME_FILENAME


def write_snapshot(cfg: Any) -> Path:
    """Dump the resolved config to `runtime.json`.

    Called once from `cli.py` at startup, before any subprocess is spawned.
    Exports `EYETOR_RUNTIME_DIR` into the current process so child processes
    inherit it via normal env inheritance.
    """
    target_dir = runtime_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EYETOR_RUNTIME_DIR", str(target_dir))

    providers: dict[str, dict[str, str]] = {}
    for name, prov in (cfg.providers or {}).items():
        providers[name] = {
            "type": prov.type,
            "base_url": prov.base_url or "",
            "model": prov.model or "",
            "api_key": prov.api_key or "",
        }

    vision: dict[str, str] | None = None
    if cfg.vision_provider:
        vp = cfg.providers.get(cfg.vision_provider)
        if vp:
            vision = {
                "provider": cfg.vision_provider,
                "base_url": vp.base_url or "",
                "model": cfg.vision_model or vp.model or "",
                "api_key": vp.api_key or "",
            }

    image: dict[str, str] | None = None
    if cfg.default_image_provider:
        image = {"provider": cfg.default_image_provider}

    knowledge_block: dict[str, Any] = {"enabled": False, "workspaces": []}
    if cfg.knowledge and cfg.knowledge.enabled:
        knowledge_block = {
            "enabled": True,
            "workspaces": [w.name for w in (cfg.knowledge.workspaces or [])],
        }

    snapshot = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "default_provider": cfg.default_provider,
        "providers": providers,
        "vision": vision,
        "image": image,
        "paths": {
            "memory_db": str(Path(cfg.memory_db_path).expanduser()),
            "tracking_db": str(Path(cfg.tracking.db_path).expanduser()),
            "sessions_dir": str(Path(cfg.sessions.dir).expanduser())
            if getattr(cfg.sessions, "dir", None)
            else "",
        },
        "knowledge": knowledge_block,
    }

    target = target_dir / RUNTIME_FILENAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    logger.info("Runtime snapshot written to %s", target)
    return target


def read_snapshot() -> dict[str, Any] | None:
    """Read the runtime snapshot. Returns None if missing or unreadable.

    Intended for use from skill subprocesses and external tools:

        from eyetor.runtime import read_snapshot
        snap = read_snapshot()
        if snap:
            url = snap["vision"]["base_url"]
    """
    path = runtime_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read runtime snapshot %s: %s", path, exc)
        return None
