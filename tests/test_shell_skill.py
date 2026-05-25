"""Tests for the generic shell skill watchdog."""

from __future__ import annotations

import importlib.util
import shlex
import sys
from pathlib import Path


def _load_shell_run_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "skills" / "shell" / "scripts" / "run.py"
    spec = importlib.util.spec_from_file_location("eyetor_shell_run", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shell_idle_timeout_kills_silent_command() -> None:
    shell_run = _load_shell_run_module()
    command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(2)'"

    result = shell_run.run_command(command, timeout=5, idle_timeout=1)

    assert result["exit_code"] == -1
    assert result["timed_out"] is True
    assert result["timeout_type"] == "idle"
    assert "without stdout/stderr output" in result["stderr"]


def test_shell_activity_prevents_idle_timeout() -> None:
    shell_run = _load_shell_run_module()
    py = shlex.quote(sys.executable)
    code = "import time; print('first', flush=True); time.sleep(0.2); print('second', flush=True)"
    command = f"{py} -c {shlex.quote(code)}"

    result = shell_run.run_command(command, timeout=5, idle_timeout=1)

    assert result["exit_code"] == 0
    assert result["stdout"] == "first\nsecond"
    assert "timed_out" not in result


def test_shell_absolute_timeout_stops_chatty_command() -> None:
    shell_run = _load_shell_run_module()
    py = shlex.quote(sys.executable)
    code = (
        "import time\n"
        "while True:\n"
        "    print('tick', flush=True)\n"
        "    time.sleep(0.1)\n"
    )
    command = f"{py} -c {shlex.quote(code)}"

    result = shell_run.run_command(command, timeout=1, idle_timeout=5)

    assert result["exit_code"] == -1
    assert result["timed_out"] is True
    assert result["timeout_type"] == "absolute"
    assert "absolute timeout" in result["stderr"]
    assert "tick" in result["stdout"]
