"""Hook execution — runs Pre/Post tool use hooks as subprocesses."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_HOOK_TIMEOUT = 10.0  # seconds


@dataclass
class HookDecision:
    """Result of running pre_tool_use hooks."""

    allow: bool = True
    deny_reason: str = ""
    modified_input: str | None = None  # JSON string of modified args
    provided_result: str | None = None  # short-circuit: return this instead of executing

    @property
    def deny(self) -> bool:
        return not self.allow


async def run_hook(
    script_path: str,
    event: str,
    tool_name: str,
    tool_input: str,
    tool_result: str = "",
    tool_error: str = "",
    tool_duration_ms: int | None = None,
) -> str:
    """Run a hook script as a subprocess with context in env vars.

    Returns stdout (JSON for pre hooks, ignored for post hooks).
    """
    env = dict(os.environ)
    env["HOOK_EVENT"] = event
    env["HOOK_TOOL_NAME"] = tool_name
    env["HOOK_TOOL_INPUT"] = tool_input
    if tool_result:
        env["HOOK_TOOL_RESULT"] = tool_result
    if tool_error:
        env["HOOK_TOOL_ERROR"] = tool_error
    if tool_duration_ms is not None:
        env["HOOK_TOOL_DURATION_MS"] = str(tool_duration_ms)

    # Determine interpreter
    if script_path.endswith(".py"):
        cmd = ["python", script_path]
    elif script_path.endswith((".sh", ".bash")):
        cmd = ["bash", script_path]
    else:
        cmd = [script_path]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_HOOK_TIMEOUT
        )
        if proc.returncode != 0:
            logger.warning(
                "Hook %s returned %d: %s", script_path, proc.returncode,
                stderr.decode(errors="replace")[:500],
            )
        return stdout.decode(errors="replace").strip()
    except asyncio.TimeoutError:
        logger.warning("Hook %s timed out after %.0fs", script_path, _HOOK_TIMEOUT)
        return ""
    except Exception as exc:
        logger.warning("Hook %s failed: %s", script_path, exc)
        return ""


def parse_pre_hook_output(stdout: str) -> HookDecision:
    """Parse the JSON output of a pre_tool_use hook into a HookDecision."""
    if not stdout:
        return HookDecision(allow=True)
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return HookDecision(allow=True)

    decision = data.get("decision", "allow")
    if decision == "deny":
        return HookDecision(allow=False, deny_reason=data.get("reason", "Denied by plugin hook"))
    if decision == "modify":
        modified = data.get("input")
        if modified is not None:
            return HookDecision(allow=True, modified_input=json.dumps(modified))
    if decision == "provide_result":
        result = data.get("result")
        if result is not None:
            provided = result if isinstance(result, str) else json.dumps(result)
            return HookDecision(allow=True, provided_result=provided)
    return HookDecision(allow=True)
