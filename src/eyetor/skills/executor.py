"""Skill script executor — runs skill scripts as subprocesses."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120.0  # seconds


async def run_script(
    script_path: Path,
    args: list[str],
    timeout: float = DEFAULT_TIMEOUT,
    cwd: Path | None = None,
) -> str:
    """Execute a skill script and return its stdout as a string.

    Args:
        script_path: Absolute path to the script.
        args: Command-line arguments to pass to the script.
        timeout: Maximum execution time in seconds.
        cwd: Working directory for the subprocess (defaults to script's parent).

    Returns:
        stdout output as string. If the script fails, returns a JSON error object.
    """
    cwd = cwd or script_path.parent
    # Choose interpreter based on extension
    if script_path.suffix == ".py":
        cmd = [sys.executable, str(script_path)] + args
    elif script_path.suffix in {".sh", ".bash"}:
        cmd = ["bash", str(script_path)] + args
    elif script_path.suffix == ".js":
        cmd = ["node", str(script_path)] + args
    else:
        cmd = [str(script_path)] + args

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            out = stdout.decode(errors="replace").strip()
            logger.warning("Script %s exited %d: %s", script_path.name, proc.returncode, err or out)
            # Prefer stdout (scripts print JSON errors there); fall back to stderr
            return out or json.dumps({"error": err or f"Script exited with code {proc.returncode}"})
        return stdout.decode(errors="replace").strip()
    except asyncio.TimeoutError:
        logger.error("Script %s timed out after %.1fs", script_path.name, timeout)
        return json.dumps({"error": f"Script timed out after {timeout}s"})
    except FileNotFoundError as e:
        logger.error("Script interpreter not found for %s: %s", script_path.name, e)
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Unexpected error running %s: %s", script_path.name, e)
        return json.dumps({"error": str(e)})
