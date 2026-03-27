#!/usr/bin/env python3
"""Execute a shell command and return its output as JSON.

Usage:
    run.py --cmd "git status"
    run.py --cmd "npm install" --cwd "/path/to/project" --timeout 60
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys


def run_command(cmd: str, cwd: str | None = None, timeout: int = 30) -> dict:
    """Execute a shell command and return structured output."""
    if platform.system() == "Windows":
        full_cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]
    else:
        full_cmd = ["bash", "-c", cmd]

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            cwd=cwd or None,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "exit_code": -1}
    except FileNotFoundError as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a shell command")
    parser.add_argument("--cmd", required=True, help="Command to execute")
    parser.add_argument("--cwd", default=None, help="Working directory")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    args = parser.parse_args()

    result = run_command(args.cmd, cwd=args.cwd, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
