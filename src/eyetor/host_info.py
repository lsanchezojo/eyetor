"""Host profile detection and persistence.

The host profile is generated during first setup and reused by the agent so it
does not have to guess the operating system or package manager from scratch.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HOST_FILENAME = "host.json"

_PACKAGE_MANAGERS = (
    "pacman",
    "paru",
    "yay",
    "apt-get",
    "dnf",
    "zypper",
    "brew",
    "apk",
    "xbps-install",
    "emerge",
    "nix-env",
)


def host_profile_path(path: str | Path | None = None) -> Path:
    """Return the path where the persistent host profile is stored."""
    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get("EYETOR_RUNTIME_DIR")
    base = Path(override).expanduser() if override else Path.home() / ".eyetor"
    return base / HOST_FILENAME


def parse_os_release(text: str) -> dict[str, str]:
    """Parse `/etc/os-release` style content."""
    data: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = _parse_os_value(value)
    return data


def detect_host_profile(
    *,
    os_release_path: str | Path = "/etc/os-release",
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Detect host OS and package managers without making changes."""
    os_data: dict[str, str] = {}
    path = Path(os_release_path)
    try:
        os_data = parse_os_release(path.read_text(encoding="utf-8"))
    except OSError:
        os_data = {}

    os_id = os_data.get("ID", "").strip().lower()
    os_like = [p.lower() for p in os_data.get("ID_LIKE", "").split() if p]
    managers = [name for name in _PACKAGE_MANAGERS if which(name)]
    preferred = choose_preferred_package_manager(
        os_id=os_id,
        os_like=os_like,
        package_managers=managers,
        platform_system=platform.system(),
    )

    profile: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "os_name": os_data.get("PRETTY_NAME") or os_data.get("NAME") or platform.system(),
        "os_id": os_id,
        "os_like": os_like,
        "package_managers": managers,
        "preferred_package_manager": preferred,
        "install_hints": build_install_hints(preferred, managers),
        "avoid_package_managers": build_avoid_package_managers(os_id, os_like, managers),
        "can_install_system_packages": False,
        "install_helper": "",
        "install_helper_command": "",
        "install_strategy": "manual",
    }
    return profile


def choose_preferred_package_manager(
    *,
    os_id: str,
    os_like: list[str],
    package_managers: list[str],
    platform_system: str,
) -> str | None:
    """Choose the safest default package manager for this host."""
    if not package_managers:
        return None
    family = {os_id, *os_like}
    if platform_system == "Darwin" and "brew" in package_managers:
        return "brew"
    if "arch" in family:
        for name in ("paru", "yay", "pacman"):
            if name in package_managers:
                return name
    if family & {"debian", "ubuntu"} and "apt-get" in package_managers:
        return "apt-get"
    if family & {"fedora", "rhel", "centos"} and "dnf" in package_managers:
        return "dnf"
    if family & {"suse", "opensuse"} and "zypper" in package_managers:
        return "zypper"
    if family & {"alpine"} and "apk" in package_managers:
        return "apk"
    return package_managers[0]


def build_install_hints(preferred: str | None, managers: list[str]) -> dict[str, str]:
    """Return command templates the agent can follow for package installs."""
    hints: dict[str, str] = {"check_binary": "command -v <binary>"}
    if not preferred:
        return hints

    templates = {
        "paru": "paru -S <package>",
        "yay": "yay -S <package>",
        "pacman": "sudo pacman -S <package>",
        "apt-get": "sudo apt-get update && sudo apt-get install -y <package>",
        "dnf": "sudo dnf install -y <package>",
        "zypper": "sudo zypper install -y <package>",
        "brew": "brew install <package>",
        "apk": "sudo apk add <package>",
        "xbps-install": "sudo xbps-install -S <package>",
        "emerge": "sudo emerge <package>",
        "nix-env": "nix-env -iA <attribute>",
    }
    if preferred in templates:
        hints["manual"] = templates[preferred]
    return hints


def build_avoid_package_managers(
    os_id: str,
    os_like: list[str],
    managers: list[str],
) -> list[str]:
    """Return managers the agent should not assume for this OS family."""
    family = {os_id, *os_like}
    avoid: list[str] = []
    if "arch" in family and "apt-get" not in managers:
        avoid.append("apt-get")
    if family & {"debian", "ubuntu"} and "pacman" not in managers:
        avoid.extend(["pacman", "paru", "yay"])
    return avoid


def ensure_host_profile(
    *,
    path: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Load the persisted host profile, generating it if missing or refreshed."""
    target = host_profile_path(path)
    if not refresh:
        existing = read_host_profile(target)
        if existing:
            profile = normalize_host_profile(existing)
            if profile != existing:
                write_host_profile(profile, target)
            return profile
    profile = detect_host_profile()
    write_host_profile(profile, target)
    return profile


def read_host_profile(path: str | Path | None = None) -> dict[str, Any] | None:
    """Read `host.json`, returning None if it is missing or invalid."""
    target = host_profile_path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def normalize_host_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults added after older `host.json` profiles were generated."""
    normalized = dict(profile)
    normalized.setdefault("schema_version", 1)
    normalized.setdefault("package_managers", [])
    normalized.setdefault("install_hints", {"check_binary": "command -v <binary>"})
    normalized.setdefault("avoid_package_managers", [])
    normalized.setdefault("can_install_system_packages", False)
    normalized.setdefault("install_helper", "")
    normalized.setdefault("install_helper_command", "")
    normalized.setdefault("install_strategy", "auto" if normalized.get("can_install_system_packages") else "manual")
    normalized.pop("install_scope", None)
    return normalized


def write_host_profile(profile: dict[str, Any], path: str | Path | None = None) -> Path:
    """Persist a host profile atomically."""
    target = host_profile_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def format_host_prompt(profile: dict[str, Any] | None) -> str:
    """Build the system prompt section that describes the current host."""
    if not profile:
        return ""

    managers = profile.get("package_managers") or []
    avoid = profile.get("avoid_package_managers") or []
    os_like = profile.get("os_like") or []
    preferred = profile.get("preferred_package_manager") or ""

    lines = [
        "## Entorno del sistema",
        f"Sistema operativo: {profile.get('os_name') or 'desconocido'}",
        f"Familia/compatibilidad: {', '.join(os_like) if os_like else 'desconocida'}",
        f"Plataforma: {profile.get('platform') or '?'} {profile.get('platform_release') or ''}".strip(),
        f"Arquitectura: {profile.get('machine') or '?'}",
        f"Gestores de paquetes disponibles: {', '.join(managers) if managers else 'ninguno detectado'}",
    ]
    if preferred:
        lines.append(f"Gestor preferido: {preferred}")
    if avoid:
        lines.append(f"No uses ni asumas estos gestores en este host: {', '.join(avoid)}")
    if profile.get("can_install_system_packages") and profile.get("install_helper_command"):
        lines.append(
            "Instalacion autonoma habilitada: usa la herramienta `install_package` "
            "para paquetes del sistema; no invoques `sudo` directamente."
        )
        lines.append(
            "Si falta una herramienta del sistema, no uses `skill_shell` con pacman, paru, yay, apt-get, dnf, zypper ni apk; "
            "llama primero a `install_package` con el nombre del paquete."
        )
        lines.append(f"Estrategia de instalacion: {profile.get('install_strategy') or 'auto'}")
        lines.append(f"Helper de instalacion: {profile['install_helper_command']} <package>")
        lines.append("El helper decide automaticamente el metodo correcto segun el sistema operativo detectado.")
    else:
        lines.append("Instalacion autonoma de paquetes del sistema: no configurada.")
    lines.append("Antes de instalar una herramienta, comprueba si ya existe con `command -v <binario>`.")
    lines.append("No cambies de gestor de paquetes salvo que la deteccion del host lo justifique.")
    return "\n".join(lines)


def _parse_os_value(value: str) -> str:
    try:
        parsed = shlex.split(value, posix=True)
    except ValueError:
        return value.strip().strip('"\'')
    return parsed[0] if parsed else ""
