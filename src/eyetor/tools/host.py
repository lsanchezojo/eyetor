"""Host tools: restricted package installs and safety guards for host skills."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console

from eyetor.models.tools import ToolDefinition, ToolRegistry

console = Console()


def register_install_tool(tool_registry: ToolRegistry, host_profile: dict) -> None:
    """Register ``install_package`` (privileged helper, no arbitrary sudo)."""
    install_package_handler = _make_install_package_handler(host_profile)
    tool_registry.register(
        ToolDefinition(
            name="install_package",
            description=(
                "Install a system package using Eyetor's restricted "
                "privileged helper. Use only after checking whether the binary "
                "already exists. The helper automatically chooses the OS-specific "
                "install strategy and does not allow arbitrary sudo commands."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "System package name to install.",
                    },
                    "binary": {
                        "type": "string",
                        "description": "Optional binary expected after install, e.g. megadl.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for installing this package.",
                    },
                },
                "required": ["package"],
            },
            handler=install_package_handler,
            group="install",
        )
    )


def _make_install_package_handler(host_profile: dict):
    async def _handler(
        package: str,
        binary: str | None = None,
        reason: str | None = None,
    ) -> str:
        import shutil

        from eyetor.install_helper import is_safe_package_name

        package = (package or "").strip()
        binary = (binary or "").strip() or None
        reason = (reason or "").strip()

        if not package or not is_safe_package_name(package):
            return json.dumps(
                {"ok": False, "error": f"Invalid package name: {package!r}"},
                ensure_ascii=False,
            )

        if binary:
            existing = shutil.which(binary)
            if existing:
                return json.dumps(
                    {
                        "ok": True,
                        "already_installed": True,
                        "binary": binary,
                        "path": existing,
                    },
                    ensure_ascii=False,
                )

        if not host_profile.get("can_install_system_packages"):
            return json.dumps(
                {
                    "ok": False,
                    "error": "Privileged install helper is not configured.",
                    "setup": "Run: sudo .venv/bin/eyetor setup --install-helper --service-user <user>",
                },
                ensure_ascii=False,
            )

        helper = str(host_profile.get("install_helper") or "")
        if not helper or not Path(helper).exists():
            return json.dumps(
                {
                    "ok": False,
                    "error": "Configured install helper is missing.",
                    "helper": helper,
                },
                ensure_ascii=False,
            )
        if not shutil.which("sudo"):
            return json.dumps(
                {"ok": False, "error": "sudo is not available on this host."},
                ensure_ascii=False,
            )

        proc = await asyncio.create_subprocess_exec(
            "sudo",
            "-n",
            helper,
            package,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=1800)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return json.dumps(
                {"ok": False, "error": "Package install timed out."},
                ensure_ascii=False,
            )

        stdout = stdout_b.decode(errors="replace").strip()
        stderr = stderr_b.decode(errors="replace").strip()
        try:
            payload = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError:
            payload = {}

        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("ok", proc.returncode == 0)
        payload.setdefault("package", package)
        if reason:
            payload.setdefault("reason", reason)
        if stderr:
            payload.setdefault("sudo_stderr", stderr[-2000:])
        if stdout and "message" not in payload:
            payload.setdefault("stdout", stdout[-2000:])
        if proc.returncode != 0:
            payload["ok"] = False
            payload.setdefault("error", "Package install helper failed.")

        if binary and payload.get("ok"):
            installed = shutil.which(binary)
            payload["binary"] = binary
            payload["binary_path"] = installed or ""
            if not installed:
                payload["ok"] = False
                payload.setdefault(
                    "error",
                    f"Package installed but expected binary {binary!r} was not found in PATH.",
                )

        return json.dumps(payload, ensure_ascii=False)

    return _handler


def _looks_like_package_install_command(args: str) -> bool:
    """Return True for shell args that should use install_package instead."""
    tokens = _split_args(args)
    if not tokens:
        return False
    if "--cmd" in tokens:
        idx = tokens.index("--cmd")
        command = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        tokens = _split_args(command)
    if tokens[:2] == ["sudo", "-n"]:
        tokens = tokens[2:]
    elif tokens and tokens[0] == "sudo":
        tokens = tokens[1:]
    if len(tokens) < 2:
        return False

    manager = tokens[0]
    if manager in {"pacman", "paru", "yay"}:
        return any(t.startswith("-S") for t in tokens[1:])
    if manager == "apt-get":
        return any(t in {"install", "update"} for t in tokens[1:])
    if manager in {"dnf", "zypper"}:
        return "install" in tokens[1:]
    if manager == "apk":
        return "add" in tokens[1:]
    return False


_SHELL_DEFAULT_ABSOLUTE_TIMEOUT = 900.0
_SHELL_MAX_ABSOLUTE_TIMEOUT = 3600.0
_SHELL_TIMEOUT_GRACE = 10.0


def _skill_execution_timeout(
    skill_name: str,
    arg_list: list[str],
    meta_timeout: float | None,
) -> float:
    """Return the outer script timeout for a routed skill invocation."""
    from eyetor.skills.executor import DEFAULT_TIMEOUT

    if skill_name != "shell":
        return meta_timeout if meta_timeout is not None else DEFAULT_TIMEOUT

    requested = _shell_requested_timeout(arg_list)
    absolute = requested if requested is not None else _SHELL_DEFAULT_ABSOLUTE_TIMEOUT
    absolute = max(1.0, min(float(absolute), _SHELL_MAX_ABSOLUTE_TIMEOUT))
    return absolute + _SHELL_TIMEOUT_GRACE


def _shell_requested_timeout(arg_list: list[str]) -> int | None:
    """Extract shell run.py's --timeout value from routed arguments."""
    for idx, token in enumerate(arg_list):
        if token == "--timeout" and idx + 1 < len(arg_list):
            try:
                return int(arg_list[idx + 1])
            except ValueError:
                return None
        if token.startswith("--timeout="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return None
    return None


_DANGEROUS_PATTERNS = [
    (
        "shell",
        "run.py",
        [
            "rm -rf",
            "rmdir /s",
            "format",
            "del /f",
            "dd if=",
            "mkfs",
            "drop table",
            "drop database",
            "> /dev/",
            "shutdown",
            "reboot",
        ],
    ),
]


def _is_dangerous(skill: str, script: str, args: str) -> bool:
    args_lower = args.lower()
    for d_skill, d_script, patterns in _DANGEROUS_PATTERNS:
        if skill == d_skill and script == d_script:
            for p in patterns:
                if p in args_lower:
                    return True
    return False


async def _ask_confirm(skill: str, script: str, args: str) -> bool:
    console.print(
        f"\n[bold red]⚠  Dangerous operation requested:[/bold red]\n"
        f"  Skill: [cyan]{skill}[/cyan]  Script: [cyan]{script}[/cyan]\n"
        f"  Args:  [yellow]{args}[/yellow]\n"
    )
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None, lambda: input("  Confirm? [y/N] ").strip().lower()
    )
    return answer in {"y", "yes"}


def _split_args(args: str) -> list[str]:
    import shlex

    try:
        return shlex.split(args)
    except ValueError:
        return args.split()
