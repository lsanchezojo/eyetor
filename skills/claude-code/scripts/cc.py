#!/usr/bin/env python3
"""Delegate a task to Claude Code (the `claude` CLI) in headless mode.

Standalone skill script (no eyetor imports). Reads its own config from
``../config.json`` (next to the skill root), runs ``claude -p`` in the resolved
repo, and prints a concise, SLM-friendly summary of the result to stdout.

Session isolation
-----------------
Each eyetor chat gets its **own** Claude Code thread per repo. We do NOT use
``--continue`` (which resumes whatever session was last touched in that directory
and would bleed across unrelated chats — or even a developer's own CC session in
the same repo). Instead we track the CC ``session_id`` we created, keyed by
``EYETOR_SESSION_ID`` (injected by eyetor) + repo path, in
``~/.eyetor/claude_code_sessions.json`` and resume it with ``--resume <id>``.

Usage:
    cc.py [repo] <task...> [--repo NAME|PATH] [--mode code|system] [--new]

Examples:
    cc.py "arregla el login" --repo odysseus
    cc.py odysseus arregla el login
    cc.py "reinicia el servicio nginx" --mode system --new
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = SKILL_ROOT / "config.json"
STATE_PATH = Path(os.path.expanduser("~/.eyetor/claude_code_sessions.json"))

SYSTEM_MODE_PROMPT = (
    "You are operating as a Linux sysadmin / ops engineer. Run diagnostics before "
    "any mutating command, mark destructive operations clearly, and end by stating "
    "exactly what you changed on the system."
)


def _fail(message: str, **extra) -> None:
    """Print a JSON error to stdout and exit non-zero."""
    payload = {"ok": False, "error": message}
    payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False))
    sys.exit(1)


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail(f"Config not found at {CONFIG_PATH}")
    except json.JSONDecodeError as exc:
        _fail(f"Invalid config.json: {exc}")
    return {}  # unreachable


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


# --- session state (per eyetor chat + repo) -------------------------------

def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, STATE_PATH)
    except OSError:
        pass  # state is best-effort; never break the run over it


def _eyetor_session() -> str:
    """Store key: chat identity, scoped to the eyetor conversation when given.

    New eyetor conversation → new key → fresh CC threads; resuming an old
    conversation restores its threads (no on_reset hook needed anymore).
    """
    sid = os.environ.get("EYETOR_SESSION_ID", "").strip() or "cli"
    conv = os.environ.get("EYETOR_CONVERSATION_ID", "").strip()
    return f"{sid}:{conv}" if conv else sid


# --- repo resolution ------------------------------------------------------

def _within_allowlist(cwd: Path, allowed: list[str]) -> bool:
    cwd_real = os.path.realpath(cwd)
    for entry in allowed:
        base = os.path.realpath(_expand(entry))
        if cwd_real == base or cwd_real.startswith(base.rstrip(os.sep) + os.sep):
            return True
    return False


def _resolve_repo(
    repo_arg: str | None,
    task_tokens: list[str],
    cfg: dict,
    session_state: dict,
) -> tuple[Path, str, list[str]]:
    """Return (cwd, repo_label, remaining_task_tokens)."""
    repos: dict[str, str] = cfg.get("repos", {}) or {}

    # 1) Explicit repo wins (from --repo or the reply token, an absolute path).
    if repo_arg:
        if repo_arg in repos:
            return _expand(repos[repo_arg]), repo_arg, task_tokens
        if repo_arg.startswith(("/", "~", ".")) or os.path.isabs(repo_arg):
            return _expand(repo_arg), _label_for_path(repo_arg, repos), task_tokens
        _fail(
            f"Unknown repo '{repo_arg}'. Known repos: {', '.join(repos) or '(none)'}. "
            "Pass a known name or an absolute/~ path."
        )

    # 2) First task token may be a known repo name.
    if task_tokens and task_tokens[0] in repos:
        name = task_tokens[0]
        return _expand(repos[name]), name, task_tokens[1:]

    # 3) No repo given: reuse this chat's last repo so follow-ups are conversational.
    last = session_state.get("last_repo")
    if last and Path(last).is_dir():
        label = _label_for_path(last, repos)
        return _expand(last), label, task_tokens

    # 4) Fall back to default_repo.
    default_repo = cfg.get("default_repo", "")
    if default_repo in repos:
        return _expand(repos[default_repo]), default_repo, task_tokens
    if default_repo:
        return _expand(default_repo), default_repo, task_tokens
    _fail("No repo specified and no valid default_repo configured.")
    raise AssertionError  # unreachable


def _label_for_path(path: str, repos: dict[str, str]) -> str:
    real = os.path.realpath(_expand(path))
    for name, p in repos.items():
        if os.path.realpath(_expand(p)) == real:
            return name
    return path


def _git_status(cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "status", "--short"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    changed = proc.stdout.strip()
    return changed or None


# --- claude invocation ----------------------------------------------------

def _build_command(
    binary: str, task: str, cwd: Path, mode: str, resume_id: str | None, cfg: dict
) -> list[str]:
    cmd = [binary, "-p", task, "--output-format", "json", "--dangerously-skip-permissions"]
    if resume_id:
        cmd += ["--resume", resume_id]
    model = cfg.get("model")
    if model:
        cmd += ["--model", str(model)]
    max_turns = cfg.get("max_turns")
    if max_turns:
        cmd += ["--max-turns", str(max_turns)]
    cmd += ["--add-dir", str(cwd)]
    if mode == "system":
        cmd += ["--append-system-prompt", SYSTEM_MODE_PROMPT]
    return cmd


def _run_claude(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def _resume_failed(proc: subprocess.CompletedProcess) -> bool:
    if proc.returncode == 0:
        return False
    blob = (proc.stdout + proc.stderr).lower()
    return (
        "no conversation" in blob
        or "no session" in blob
        or "session id" in blob
        or "not found" in blob
        or "no previous" in blob
    )


def main() -> None:
    cfg = _load_config()

    parser = argparse.ArgumentParser(description="Delegate a task to Claude Code.")
    parser.add_argument("task", nargs="*", help="Task text (first token may be a repo name).")
    parser.add_argument("--repo", default=None, help="Repo name (from config) or absolute/~ path.")
    parser.add_argument("--mode", choices=["code", "system"], default="code")
    parser.add_argument("--new", action="store_true", help="Start a fresh CC session (ignore stored one).")
    parser.add_argument("--session-reset", dest="session_reset", action="store_true",
                        help="Forget this chat's CC sessions and exit (eyetor on_reset hook).")
    args = parser.parse_args()

    state = _load_state()
    eyetor_sid = _eyetor_session()

    # eyetor /reset: drop this chat's stored CC sessions so the next task starts fresh.
    if args.session_reset:
        if state.pop(eyetor_sid, None) is not None:
            _save_state(state)
        print(json.dumps({"ok": True, "message": f"Claude Code sessions cleared for {eyetor_sid}."}, ensure_ascii=False))
        sys.exit(0)

    sess = state.setdefault(eyetor_sid, {"last_repo": None, "repos": {}})

    # A reply-to-continue invocation hands back the repo path via EYETOR_REPLY_TOKEN,
    # which acts exactly like an explicit --repo (continues that message's thread).
    repo_arg = args.repo or os.environ.get("EYETOR_REPLY_TOKEN") or None
    cwd, repo_label, task_tokens = _resolve_repo(repo_arg, list(args.task), cfg, sess)
    task = " ".join(task_tokens).strip()
    if not task:
        _fail("No task provided. Usage: cc.py [repo] <task...> [--mode system] [--new]")

    if not cwd.is_dir():
        _fail(f"Working directory does not exist: {cwd}")
    if not _within_allowlist(cwd, cfg.get("allowed_dirs", []) or []):
        _fail(f"Directory {cwd} is outside the allowed_dirs allowlist.")

    binary = cfg.get("binary", "claude")
    timeout = float(cfg.get("claude_timeout_seconds", 1700))
    cwd_key = str(cwd)

    # Which CC session to resume (ours only — never a stray directory session).
    resume_id = None if args.new else sess.get("repos", {}).get(cwd_key)

    started = time.monotonic()
    cmd = _build_command(binary, task, cwd, args.mode, resume_id, cfg)
    try:
        proc = _run_claude(cmd, cwd, timeout)
        # Stored session gone/invalid → retry fresh.
        if resume_id and _resume_failed(proc):
            cmd = _build_command(binary, task, cwd, args.mode, None, cfg)
            proc = _run_claude(cmd, cwd, timeout)
    except FileNotFoundError:
        _fail(f"Claude binary not found: {binary}")
    except subprocess.TimeoutExpired:
        _fail(f"Claude Code timed out after {timeout:.0f}s in {cwd}.", repo=repo_label)
        return

    elapsed = time.monotonic() - started

    raw = proc.stdout.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        err = proc.stderr.strip() or raw or f"exit code {proc.returncode}"
        _fail(f"Could not parse Claude Code output: {err[:1500]}", repo=repo_label)
        return

    result_text = (data.get("result") or "").strip()
    is_error = bool(data.get("is_error"))
    session_id = data.get("session_id", "")
    num_turns = data.get("num_turns")
    cost = data.get("total_cost_usd")

    # Persist the CC session so the next turn in this chat+repo resumes it.
    if session_id and not is_error:
        sess.setdefault("repos", {})[cwd_key] = session_id
        sess["last_repo"] = cwd_key
        _save_state(state)

    max_chars = int(cfg.get("output_max_chars", 4000))
    reserved = 400  # room for the metadata footer
    if len(result_text) > max_chars - reserved:
        result_text = result_text[: max_chars - reserved].rstrip() + "\n…[truncado]"

    lines = [f"🤖 Claude Code · repo={repo_label} · modo={args.mode}" + (" · ⚠ error" if is_error else "")]
    lines.append("")
    lines.append(result_text or "(sin texto de resultado)")

    changed = _git_status(cwd)
    if changed:
        lines.append("")
        lines.append("Cambios sin commitear (git status --short):")
        lines.append(changed[:800])

    meta = []
    if num_turns is not None:
        meta.append(f"{num_turns} turns")
    meta.append(f"{elapsed:.0f}s")
    if isinstance(cost, (int, float)):
        meta.append(f"${cost:.4f}")
    if session_id:
        meta.append(f"sesión {session_id[:8]}")
    lines.append("")
    lines.append("· ".join(meta))

    out = {"message": "\n".join(lines)}
    # Opaque continuation handle for eyetor's reply-to-continue: the repo path.
    # Only when a real thread was created, so replies to errors aren't tracked.
    if session_id and not is_error:
        out["reply_token"] = cwd_key
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(1 if is_error else 0)


if __name__ == "__main__":
    main()
