"""Plugin discovery — scans directories for plugin.json files."""

from __future__ import annotations

import logging
from pathlib import Path

from eyetor.plugins.manifest import PluginManifest, load_manifest

logger = logging.getLogger(__name__)


def discover_plugins(plugin_dirs: list[str]) -> list[PluginManifest]:
    """Scan directories for valid plugin.json files.

    Later directories override earlier ones (same name).
    """
    found: dict[str, PluginManifest] = {}
    for dir_str in plugin_dirs:
        base = Path(dir_str).expanduser()
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            manifest = load_manifest(child)
            if manifest:
                found[manifest.name] = manifest
    logger.info("Discovered %d plugins from %s", len(found), plugin_dirs)
    return list(found.values())
