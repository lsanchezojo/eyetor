#!/usr/bin/env python3
"""Execute a shell command and return its output as JSON.

Usage:
    run.py --cmd "git status"
    run.py --cmd "npm install" --cwd "/path/to/project" --timeout 900 --idle-timeout 120
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import signal
import subprocess
import sys
import threading
import time

_MAX_OUTPUT_BYTES = 64_000


def run_command(
    cmd: str,
    cwd: str | None = None,
    timeout: int = 900,
    idle_timeout: int = 120,
) -> dict:
    """Execute a shell command and return structured output."""
    if platform.system() == "Windows":
        full_cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]
        popen_kwargs = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    else:
        full_cmd = ["bash", "-c", cmd]
        popen_kwargs = {"preexec_fn": os.setsid}

    started = time.monotonic()
    last_output = started
    stdout_b = b""
    stderr_b = b""
    events: queue.Queue[tuple[str, bytes]] = queue.Queue()

    try:
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd or None,
            **popen_kwargs,
        )

        threads = [
            threading.Thread(
                target=_read_stream,
                args=(proc.stdout, "stdout", events),
                daemon=True,
            ),
            threading.Thread(
                target=_read_stream,
                args=(proc.stderr, "stderr", events),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        while True:
            drained = False
            try:
                while True:
                    stream, chunk = events.get_nowait()
                    drained = True
                    last_output = time.monotonic()
                    if stream == "stdout":
                        stdout_b = _append_tail(stdout_b, chunk)
                    else:
                        stderr_b = _append_tail(stderr_b, chunk)
            except queue.Empty:
                pass

            returncode = proc.poll()
            if returncode is not None:
                for thread in threads:
                    thread.join(timeout=0.2)
                stdout_b, stderr_b = _drain_events(events, stdout_b, stderr_b)
                return _result(stdout_b, stderr_b, returncode, started)

            now = time.monotonic()
            timeout_type = ""
            if timeout > 0 and now - started >= timeout:
                timeout_type = "absolute"
            elif idle_timeout > 0 and now - last_output >= idle_timeout:
                timeout_type = "idle"

            if timeout_type:
                _terminate_process(proc)
                for thread in threads:
                    thread.join(timeout=0.5)
                stdout_b, stderr_b = _drain_events(events, stdout_b, stderr_b)
                message = _timeout_message(timeout_type, timeout, idle_timeout)
                stderr_b = _append_tail(stderr_b, message.encode("utf-8"))
                payload = _result(stdout_b, stderr_b, -1, started)
                payload["timeout_type"] = timeout_type
                payload["timed_out"] = True
                return payload

            if not drained:
                time.sleep(0.05)

    except FileNotFoundError as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}


def _read_stream(pipe, stream: str, events: queue.Queue[tuple[str, bytes]]) -> None:
    if pipe is None:
        return
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                return
            events.put((stream, chunk))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _append_tail(current: bytes, chunk: bytes) -> bytes:
    combined = current + chunk
    if len(combined) <= _MAX_OUTPUT_BYTES:
        return combined
    return combined[-_MAX_OUTPUT_BYTES:]


def _drain_events(
    events: queue.Queue[tuple[str, bytes]],
    stdout_b: bytes,
    stderr_b: bytes,
) -> tuple[bytes, bytes]:
    try:
        while True:
            stream, chunk = events.get_nowait()
            if stream == "stdout":
                stdout_b = _append_tail(stdout_b, chunk)
            else:
                stderr_b = _append_tail(stderr_b, chunk)
    except queue.Empty:
        return stdout_b, stderr_b


def _terminate_process(proc: subprocess.Popen) -> None:
    try:
        if platform.system() == "Windows":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=3)
    except Exception:
        try:
            if platform.system() == "Windows":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _result(stdout_b: bytes, stderr_b: bytes, exit_code: int, started: float) -> dict:
    return {
        "stdout": stdout_b.decode(errors="replace").strip(),
        "stderr": stderr_b.decode(errors="replace").strip(),
        "exit_code": exit_code,
        "runtime_seconds": round(time.monotonic() - started, 3),
    }


def _timeout_message(timeout_type: str, timeout: int, idle_timeout: int) -> str:
    if timeout_type == "idle":
        return (
            f"\nCommand stopped after {idle_timeout}s without stdout/stderr output. "
            "If this command is expected to be silent for longer, retry with "
            "--idle-timeout N outside --cmd."
        )
    return (
        f"\nCommand stopped after the absolute timeout of {timeout}s. "
        "For long-running downloads/builds, retry with a larger --timeout N outside --cmd."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a shell command")
    parser.add_argument("--cmd", default=None, help="Command to execute")
    parser.add_argument("--cwd", default=None, help="Working directory")
    parser.add_argument("--timeout", type=int, default=900, help="Absolute timeout in seconds")
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=120,
        help="Stop after this many seconds without stdout/stderr output",
    )
    parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()

    cmd = args.cmd
    if not cmd and args.rest:
        # Small models often forget the --cmd flag and pass the command
        # as positional tokens. Re-join them into a single shell string.
        cmd = " ".join(args.rest)
    if not cmd:
        parser.error("a command is required (use --cmd \"...\" or pass it as positional args)")

    result = run_command(
        cmd,
        cwd=args.cwd,
        timeout=args.timeout,
        idle_timeout=args.idle_timeout,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
