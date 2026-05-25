"""Install a restricted privileged helper for system package installs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from eyetor.host_info import write_host_profile

HELPER_PATH = Path("/usr/local/sbin/eyetor-install-tool")
SUDOERS_PATH = Path("/etc/sudoers.d/eyetor-install-tool")

_SERVICE_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}\$?$")
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@:-]{0,127}$")

HELPER_SCRIPT = r'''#!/usr/bin/python3
"""Restricted package installer for Eyetor.

This script is intended to be installed as root-owned and invoked through a
narrow sudoers rule. It accepts only package names, never arbitrary shell
commands or package-manager flags. The install strategy is automatic and based
on the detected operating system.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@:-]{0,127}$")
LOG_PATH = Path("/var/log/eyetor-install-tool.log")
BUILD_ROOT = Path("/var/tmp/eyetor-install-tool")
ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LC_ALL": "C.UTF-8",
}


def main() -> int:
    if os.geteuid() != 0:
        return _emit(False, "helper must run as root", exit_code=1)

    packages = sys.argv[1:]
    if not packages:
        return _emit(False, "at least one package name is required", exit_code=2)
    invalid = [p for p in packages if not PACKAGE_RE.fullmatch(p)]
    if invalid:
        return _emit(False, f"invalid package name(s): {', '.join(invalid)}", exit_code=2)

    strategy = _detect_strategy()
    if not strategy:
        return _emit(False, "no supported package strategy detected", exit_code=3)

    if strategy == "arch-family":
        return _install_arch_family(packages)

    commands = _commands_for(strategy, packages)
    result = _run_commands(commands)
    if not result["ok"]:
        details = {k: v for k, v in result.items() if k not in {"ok", "message"}}
        return _emit(
            False,
            "package manager failed",
            strategy=strategy,
            packages=packages,
            **details,
            exit_code=result["returncode"] or 1,
        )

    return _emit(
        True,
        "packages installed",
        strategy=strategy,
        packages=packages,
        stdout=_tail(result["stdout"]),
        stderr=_tail(result["stderr"]),
        exit_code=0,
    )


def _detect_strategy() -> str | None:
    os_id, os_like = _os_family()
    family = {os_id, *os_like}
    if "arch" in family and shutil.which("pacman"):
        return "arch-family"
    if family & {"debian", "ubuntu"} and shutil.which("apt-get"):
        return "apt-get"
    if family & {"fedora", "rhel", "centos"} and shutil.which("dnf"):
        return "dnf"
    if family & {"suse", "opensuse"} and shutil.which("zypper"):
        return "zypper"
    if "alpine" in family and shutil.which("apk"):
        return "apk"
    return None


def _commands_for(strategy: str, packages: list[str]) -> list[list[str]]:
    if strategy == "apt-get":
        return [
            ["apt-get", "update"],
            ["apt-get", "install", "-y", "--no-install-recommends", *packages],
        ]
    if strategy == "dnf":
        return [["dnf", "install", "-y", *packages]]
    if strategy == "zypper":
        return [["zypper", "--non-interactive", "install", *packages]]
    if strategy == "apk":
        return [["apk", "add", *packages]]
    raise ValueError(f"unsupported strategy: {strategy}")


def _install_arch_family(packages: list[str]) -> int:
    official: list[str] = []
    community: list[str] = []
    for package in packages:
        if _pacman_has_package(package):
            official.append(package)
        else:
            community.append(package)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    installed: list[dict[str, str]] = []

    if official:
        result = _run_commands([["pacman", "-S", "--needed", "--noconfirm", *official]])
        stdout_parts.append(result["stdout"])
        stderr_parts.append(result["stderr"])
        if not result["ok"]:
            return _emit(
                False,
                "official package install failed",
                strategy="arch-family",
                packages=packages,
                official=official,
                returncode=result["returncode"],
                stdout=_tail("".join(stdout_parts)),
                stderr=_tail("".join(stderr_parts)),
                exit_code=result["returncode"] or 1,
            )
        installed.extend({"package": p, "source": "system"} for p in official)

    for package in community:
        result = _install_arch_community_package(package)
        stdout_parts.append(result["stdout"])
        stderr_parts.append(result["stderr"])
        if not result["ok"]:
            return _emit(
                False,
                "community package install failed",
                strategy="arch-family",
                packages=packages,
                package=package,
                stage=result.get("stage", "unknown"),
                returncode=result["returncode"],
                stdout=_tail("".join(stdout_parts)),
                stderr=_tail("".join(stderr_parts)),
                exit_code=result["returncode"] or 1,
            )
        installed.append({"package": package, "source": "community"})

    return _emit(
        True,
        "packages installed",
        strategy="arch-family",
        packages=packages,
        installed=installed,
        stdout=_tail("".join(stdout_parts)),
        stderr=_tail("".join(stderr_parts)),
        exit_code=0,
    )


def _install_arch_community_package(package: str) -> dict:
    service_user = _service_user()
    if not service_user:
        return _result(False, "missing SUDO_USER for community build", returncode=4, stage="service_user")

    result = _ensure_arch_build_tools()
    if not result["ok"]:
        result["stage"] = "build_tools"
        return result

    checkout = _prepare_build_dir(package, service_user)
    if isinstance(checkout, dict):
        return checkout

    clone = _run_as_user(
        service_user,
        ["git", "clone", "--depth=1", f"https://aur.archlinux.org/{package}.git", str(checkout)],
        timeout=300,
    )
    if not clone["ok"]:
        clone["stage"] = "fetch"
        return clone

    deps = _read_srcinfo_dependencies(checkout / ".SRCINFO")
    official_deps = [dep for dep in deps if _pacman_has_package(dep)]
    dep_result = {"ok": True, "stdout": "", "stderr": "", "returncode": 0}
    if official_deps:
        dep_result = _run_commands([["pacman", "-S", "--needed", "--noconfirm", *official_deps]])
        if not dep_result["ok"]:
            dep_result["stage"] = "dependencies"
            return dep_result

    build = _run_as_user(
        service_user,
        ["makepkg", "-f", "--noconfirm", "--noprogressbar", "--skippgpcheck"],
        timeout=3600,
        cwd=checkout,
    )
    if not build["ok"]:
        build["stage"] = "build"
        build["stdout"] = dep_result["stdout"] + clone["stdout"] + build["stdout"]
        build["stderr"] = dep_result["stderr"] + clone["stderr"] + build["stderr"]
        return build

    artifacts = sorted(checkout.glob("*.pkg.tar.*"))
    if not artifacts:
        return _result(False, "no built package artifact found", returncode=5, stage="artifact")

    install = _run_commands([["pacman", "-U", "--noconfirm", *[str(p) for p in artifacts]]])
    install["stdout"] = dep_result["stdout"] + clone["stdout"] + build["stdout"] + install["stdout"]
    install["stderr"] = dep_result["stderr"] + clone["stderr"] + build["stderr"] + install["stderr"]
    if not install["ok"]:
        install["stage"] = "install"
    return install


def _ensure_arch_build_tools() -> dict:
    return _run_commands([["pacman", "-S", "--needed", "--noconfirm", "git", "base-devel"]])


def _prepare_build_dir(package: str, service_user: str) -> Path | dict:
    import pwd

    try:
        info = pwd.getpwnam(service_user)
    except KeyError:
        return _result(False, f"unknown service user: {service_user}", returncode=4, stage="service_user")
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    os.chown(BUILD_ROOT, info.pw_uid, info.pw_gid)
    checkout = BUILD_ROOT / package
    shutil.rmtree(checkout, ignore_errors=True)
    return checkout


def _read_srcinfo_dependencies(path: Path) -> list[str]:
    deps: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    keys = {"depends", "makedepends", "checkdepends"}
    for line in lines:
        stripped = line.strip()
        if " = " not in stripped:
            continue
        key, value = stripped.split(" = ", 1)
        if key not in keys:
            continue
        dep = re.split(r"[<>=]", value, 1)[0].strip()
        if PACKAGE_RE.fullmatch(dep):
            deps.add(dep)
    return sorted(deps)


def _pacman_has_package(package: str) -> bool:
    result = subprocess.run(
        ["pacman", "-Si", package],
        capture_output=True,
        text=True,
        timeout=60,
        env=ENV,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0


def _run_commands(commands: list[list[str]]) -> dict:
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    last_returncode = 0
    for cmd in commands:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            env=ENV,
            encoding="utf-8",
            errors="replace",
        )
        stdout_parts.append(result.stdout)
        stderr_parts.append(result.stderr)
        last_returncode = result.returncode
        if result.returncode != 0:
            return _result(False, "command failed", last_returncode, stdout_parts, stderr_parts, command=cmd)
    return _result(True, "ok", last_returncode, stdout_parts, stderr_parts)


def _run_as_user(service_user: str, cmd: list[str], timeout: int, cwd: Path | None = None) -> dict:
    runuser = shutil.which("runuser")
    if not runuser:
        return _result(False, "runuser not found", returncode=4, stage="run_as_user")
    result = subprocess.run(
        [runuser, "-u", service_user, "--", *cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        env=ENV,
        encoding="utf-8",
        errors="replace",
    )
    return _result(
        result.returncode == 0,
        "ok" if result.returncode == 0 else "command failed",
        result.returncode,
        [result.stdout],
        [result.stderr],
        command=cmd,
    )


def _result(
    ok: bool,
    message: str,
    returncode: int,
    stdout_parts: list[str] | None = None,
    stderr_parts: list[str] | None = None,
    **extra,
) -> dict:
    return {
        "ok": ok,
        "message": message,
        "returncode": returncode,
        "stdout": "".join(stdout_parts or []),
        "stderr": "".join(stderr_parts or []),
        **extra,
    }


def _service_user() -> str:
    user = os.environ.get("SUDO_USER") or ""
    return "" if user in {"", "root"} else user


def _os_family() -> tuple[str, set[str]]:
    data: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key] = value.strip().strip('"')
    except OSError:
        pass
    os_id = data.get("ID", "").lower()
    os_like = {p.lower() for p in data.get("ID_LIKE", "").split() if p}
    return os_id, os_like


def _emit(ok: bool, message: str, exit_code: int, **extra) -> int:
    payload = {"ok": ok, "message": message, **extra}
    print(json.dumps(payload, ensure_ascii=False))
    _log(payload)
    return exit_code


def _log(payload: dict) -> None:
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv[1:],
            **payload,
        }
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _tail(text: str, limit: int = 6000) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


if __name__ == "__main__":
    raise SystemExit(main())
'''


def validate_service_user(user: str) -> str:
    """Validate and return a Linux service user name."""
    if not _SERVICE_USER_RE.fullmatch(user):
        raise ValueError(f"Invalid service user: {user!r}")
    return user


def is_safe_package_name(package: str) -> bool:
    """Return True if *package* is safe to pass to the install helper."""
    return bool(_PACKAGE_RE.fullmatch(package))


def render_sudoers(service_user: str, helper_path: Path = HELPER_PATH) -> str:
    """Render the narrow sudoers rule for the helper."""
    user = validate_service_user(service_user)
    return f"{user} ALL=(root) NOPASSWD: {helper_path} *\n"


def install_privileged_helper(
    *,
    service_user: str,
    host_profile: dict[str, Any],
    host_path: str | Path,
    helper_path: Path = HELPER_PATH,
    sudoers_path: Path = SUDOERS_PATH,
    require_root: bool = True,
    validate_sudoers: bool = True,
) -> dict[str, Any]:
    """Install the root-owned helper and update the persisted host profile."""
    service_user = validate_service_user(service_user)
    if require_root and os.geteuid() != 0:
        raise PermissionError("setup --install-helper must be run as root")

    _write_helper_script(helper_path)
    _write_sudoers(
        service_user=service_user,
        helper_path=helper_path,
        sudoers_path=sudoers_path,
        validate=validate_sudoers,
    )

    profile = dict(host_profile)
    profile.pop("install_scope", None)
    profile.update(
        {
            "can_install_system_packages": True,
            "install_helper": str(helper_path),
            "install_helper_command": f"sudo -n {helper_path}",
            "install_strategy": "auto",
        }
    )
    host_target = Path(host_path)
    parent_existed = host_target.parent.exists()
    target = write_host_profile(profile, host_target)
    if not parent_existed:
        _chown_to_user(host_target.parent, service_user)
    _chown_to_user(target, service_user)
    return profile


def _write_helper_script(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(HELPER_SCRIPT, encoding="utf-8")
    os.chmod(tmp, 0o755)
    if os.geteuid() == 0:
        os.chown(tmp, 0, 0)
    tmp.replace(path)
    os.chmod(path, 0o755)


def _write_sudoers(
    *,
    service_user: str,
    helper_path: Path,
    sudoers_path: Path,
    validate: bool,
) -> None:
    sudoers_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sudoers_path.with_suffix(".tmp")
    tmp.write_text(render_sudoers(service_user, helper_path), encoding="utf-8")
    os.chmod(tmp, 0o440)
    if os.geteuid() == 0:
        os.chown(tmp, 0, 0)
    if validate:
        visudo = shutil.which("visudo")
        if not visudo:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("visudo not found; refusing to install sudoers rule")
        subprocess.run([visudo, "-cf", str(tmp)], check=True)
    tmp.replace(sudoers_path)
    os.chmod(sudoers_path, 0o440)


def _chown_to_user(path: Path, service_user: str) -> None:
    if os.geteuid() != 0:
        return
    import pwd

    try:
        info = pwd.getpwnam(service_user)
    except KeyError:
        return
    os.chown(path, info.pw_uid, info.pw_gid)
