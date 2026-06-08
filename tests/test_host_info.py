"""Tests for persistent host profile detection."""

from __future__ import annotations

import json
import os
from pathlib import Path

from click.testing import CliRunner

from eyetor.cli import (
    _looks_like_package_install_command,
    _make_install_package_handler,
    _shell_requested_timeout,
    _skill_execution_timeout,
    cli,
)
from eyetor.config import VectorConfig
from eyetor.host_info import (
    detect_host_profile,
    format_host_prompt,
    ensure_host_profile,
    normalize_host_profile,
    parse_os_release,
    read_host_profile,
    write_host_profile,
)
from eyetor.install_helper import (
    install_privileged_helper,
    is_safe_package_name,
    render_sudoers,
)
from eyetor.runtime import write_snapshot


def test_parse_os_release_handles_quotes() -> None:
    data = parse_os_release('PRETTY_NAME="CachyOS"\nID=cachyos\nID_LIKE="arch"')

    assert data["PRETTY_NAME"] == "CachyOS"
    assert data["ID"] == "cachyos"
    assert data["ID_LIKE"] == "arch"


def test_detect_arch_like_host_prefers_available_native_helper(tmp_path: Path) -> None:
    os_release = tmp_path / "os-release"
    os_release.write_text(
        'PRETTY_NAME="CachyOS"\nID=cachyos\nID_LIKE=arch\n',
        encoding="utf-8",
    )
    available = {"pacman", "paru", "yay"}

    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in available else None

    profile = detect_host_profile(os_release_path=os_release, which=which)

    assert profile["os_name"] == "CachyOS"
    assert profile["os_id"] == "cachyos"
    assert profile["os_like"] == ["arch"]
    assert profile["package_managers"] == ["pacman", "paru", "yay"]
    assert profile["preferred_package_manager"] == "paru"
    assert profile["install_hints"]["manual"] == "paru -S <package>"
    assert "apt-get" in profile["avoid_package_managers"]


def test_host_prompt_tells_agent_not_to_assume_wrong_manager() -> None:
    prompt = format_host_prompt(
        {
            "os_name": "CachyOS",
            "os_like": ["arch"],
            "platform": "Linux",
            "platform_release": "7.0",
            "machine": "x86_64",
            "package_managers": ["pacman", "paru"],
            "preferred_package_manager": "paru",
            "install_hints": {"manual": "paru -S <package>"},
            "avoid_package_managers": ["apt-get"],
        }
    )

    assert "Operating system: CachyOS" in prompt
    assert "Preferred manager: paru" in prompt
    assert "Do not use or assume these managers" in prompt
    assert "apt-get" in prompt
    assert "command -v" in prompt


def test_host_prompt_mentions_install_package_when_helper_enabled() -> None:
    prompt = format_host_prompt(
        {
            "os_name": "CachyOS",
            "os_like": ["arch"],
            "platform": "Linux",
            "platform_release": "7.0",
            "machine": "x86_64",
            "package_managers": ["pacman"],
            "preferred_package_manager": "pacman",
            "install_hints": {"manual": "sudo pacman -S <package>"},
            "avoid_package_managers": ["apt-get"],
            "can_install_system_packages": True,
            "install_helper_command": "sudo -n /usr/local/sbin/eyetor-install-tool",
            "install_strategy": "auto",
        }
    )

    assert "use the `install_package` tool" in prompt
    assert "do not invoke `sudo` directly" in prompt
    assert "Install strategy: auto" in prompt
    assert "automatically picks" in prompt


def test_host_profile_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "host.json"
    profile = {"schema_version": 1, "os_name": "TestOS"}

    write_host_profile(profile, target)

    assert read_host_profile(target) == profile


def test_normalize_host_profile_adds_install_defaults() -> None:
    profile = normalize_host_profile({"os_name": "OldOS"})

    assert profile["can_install_system_packages"] is False
    assert profile["install_helper"] == ""
    assert profile["install_strategy"] == "manual"


def test_ensure_host_profile_migrates_existing_profile(tmp_path: Path) -> None:
    target = tmp_path / "host.json"
    target.write_text('{"os_name":"OldOS"}', encoding="utf-8")

    profile = ensure_host_profile(path=target)

    assert profile["can_install_system_packages"] is False
    assert json.loads(target.read_text(encoding="utf-8"))["install_strategy"] == "manual"


def test_runtime_snapshot_includes_host_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EYETOR_RUNTIME_DIR", str(tmp_path))
    host = {
        "os_name": "TestOS",
        "os_id": "test",
        "package_managers": ["testpm"],
    }

    target = write_snapshot(VectorConfig(), host_profile=host)
    data = json.loads(target.read_text(encoding="utf-8"))

    assert data["host"] == host


def test_setup_command_creates_host_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EYETOR_RUNTIME_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["setup", "--print-json"])

    assert result.exit_code == 0
    assert (tmp_path / "host.json").exists()
    data = json.loads(result.output)
    assert "os_name" in data
    assert "package_managers" in data
    assert data["can_install_system_packages"] is False


def test_package_name_validation() -> None:
    assert is_safe_package_name("megatools")
    assert is_safe_package_name("python-foo_bar+1.0")
    assert not is_safe_package_name("--noconfirm")
    assert not is_safe_package_name("foo;rm-rf")


def test_render_sudoers_is_narrow() -> None:
    assert render_sudoers("haziel") == (
        "haziel ALL=(root) NOPASSWD: /usr/local/sbin/eyetor-install-tool *\n"
    )


def test_install_privileged_helper_writes_temp_paths(tmp_path: Path) -> None:
    helper = tmp_path / "sbin" / "eyetor-install-tool"
    sudoers = tmp_path / "sudoers.d" / "eyetor-install-tool"
    host_path = tmp_path / "host.json"

    profile = install_privileged_helper(
        service_user="haziel",
        host_profile={"os_name": "TestOS"},
        host_path=host_path,
        helper_path=helper,
        sudoers_path=sudoers,
        require_root=False,
        validate_sudoers=False,
    )

    assert helper.exists()
    assert os.access(helper, os.X_OK)
    assert sudoers.read_text(encoding="utf-8") == (
        f"haziel ALL=(root) NOPASSWD: {helper} *\n"
    )
    assert profile["can_install_system_packages"] is True
    assert profile["install_strategy"] == "auto"
    assert "install_scope" not in profile
    assert "makepkg" in helper.read_text(encoding="utf-8")
    assert json.loads(host_path.read_text(encoding="utf-8"))["install_helper"] == str(helper)


def test_install_package_handler_requires_helper() -> None:
    handler = _make_install_package_handler({"can_install_system_packages": False})

    import asyncio

    result = json.loads(asyncio.run(handler(package="megatools", binary="unlikely-binary")))

    assert result["ok"] is False
    assert "not configured" in result["error"]


def test_shell_package_install_commands_are_detected() -> None:
    assert _looks_like_package_install_command('--cmd "paru -S megatools --noconfirm"')
    assert _looks_like_package_install_command('--cmd "sudo pacman -S megatools"')
    assert _looks_like_package_install_command('--cmd "apt-get install -y megatools"')
    assert not _looks_like_package_install_command('--cmd "command -v megadl"')


def test_shell_requested_timeout_is_extracted() -> None:
    assert _shell_requested_timeout(["--cmd", "sleep 1", "--timeout", "600"]) == 600
    assert _shell_requested_timeout(["--cmd", "sleep 1", "--timeout=300"]) == 300
    assert _shell_requested_timeout(["--cmd", "sleep 1"]) is None
    assert _shell_requested_timeout(["--cmd", "sleep 1", "--timeout", "bad"]) is None


def test_shell_execution_timeout_respects_requested_timeout() -> None:
    assert _skill_execution_timeout("shell", ["--cmd", "date"], None) == 910.0
    assert _skill_execution_timeout("shell", ["--cmd", "date", "--timeout", "600"], None) == 610.0
    assert _skill_execution_timeout("shell", ["--cmd", "date", "--timeout", "9999"], None) == 3610.0
    assert _skill_execution_timeout("browser", [], 42.0) == 42.0


def test_setup_install_helper_non_root_prints_bootstrap_command(tmp_path: Path, monkeypatch) -> None:
    import eyetor.cli as cli_module

    monkeypatch.setattr(cli_module.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        cli_module,
        "_host_profile_path_for_user",
        lambda _user: tmp_path / "host.json",
    )

    result = CliRunner().invoke(
        cli,
        ["setup", "--install-helper", "--service-user", "haziel"],
    )

    assert result.exit_code == 1
    assert "sudo" in result.output
    assert "--install-helper" in result.output
