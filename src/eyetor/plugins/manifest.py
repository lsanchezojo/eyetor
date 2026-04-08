"""Plugin manifest — definition and validation of plugin.json files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_PERMISSIONS = {"read_files", "write_files", "execute_shell", "network"}
VALID_HOOK_TYPES = {"pre_tool_use", "post_tool_use", "post_tool_use_failure"}


@dataclass
class PluginManifest:
    """Parsed and validated representation of a plugin.json file."""

    name: str
    version: str
    description: str
    path: Path  # directory containing plugin.json
    permissions: list[str] = field(default_factory=list)
    hooks: dict[str, str] = field(default_factory=dict)  # hook_type → script path
    lifecycle_init: str | None = None
    lifecycle_shutdown: str | None = None


def load_manifest(plugin_dir: Path) -> PluginManifest | None:
    """Load and validate a plugin.json from the given directory.

    Returns None if the file is missing or invalid.
    """
    manifest_path = plugin_dir / "plugin.json"
    if not manifest_path.exists():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse %s: %s", manifest_path, exc)
        return None

    # Required fields
    name = data.get("name")
    version = data.get("version", "0.0.0")
    description = data.get("description", "")
    if not name:
        logger.warning("Plugin at %s has no 'name' field", plugin_dir)
        return None

    # Validate name matches directory
    if name != plugin_dir.name:
        logger.warning(
            "Plugin name '%s' does not match directory '%s'", name, plugin_dir.name
        )

    # Permissions
    permissions = data.get("permissions", [])
    unknown = set(permissions) - VALID_PERMISSIONS
    if unknown:
        logger.warning("Plugin '%s' declares unknown permissions: %s", name, unknown)

    # Hooks
    hooks: dict[str, str] = {}
    raw_hooks = data.get("hooks", {})
    for hook_type, script_path in raw_hooks.items():
        if hook_type not in VALID_HOOK_TYPES:
            logger.warning("Plugin '%s' declares unknown hook type: %s", name, hook_type)
            continue
        full_path = plugin_dir / script_path
        if not full_path.exists():
            logger.warning("Plugin '%s' hook script not found: %s", name, full_path)
            continue
        hooks[hook_type] = str(full_path)

    # Lifecycle
    lifecycle = data.get("lifecycle", {})
    init_cmd = lifecycle.get("init")
    shutdown_cmd = lifecycle.get("shutdown")

    logger.info(
        "Loaded plugin '%s' v%s — permissions: %s, hooks: %s",
        name, version, permissions, list(hooks.keys()),
    )

    return PluginManifest(
        name=name,
        version=version,
        description=description,
        path=plugin_dir,
        permissions=permissions,
        hooks=hooks,
        lifecycle_init=init_cmd,
        lifecycle_shutdown=shutdown_cmd,
    )
