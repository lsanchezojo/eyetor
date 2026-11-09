"""Tool-choice eval: does the model pick the right tool for real requests?

Each case's message goes through the same system prompt, access policy and
tool gating a real session uses; the model is asked ONCE and its first
response is checked. No tool is ever executed, so cases can safely say
"apaga la luz" or "borra el fichero".

Usage (from the repo root):

    .venv/bin/python evals/tool_choice.py [CASES.yaml] [-p PROVIDER] [-n N] [-k TEXT]

CASES defaults to ~/.eyetor/evals/tool_choice.yaml (your own requests, kept
out of the repo), falling back to evals/tool_choice.example.yaml.

Case format (YAML list):

    - input: "apunta leche en la lista"
      expect: shopping_list_add          # a tool, a list of tools, or "none"
      args: {text: leche}                # optional: case-insensitive substrings
      history:                           # optional earlier turns
        - {user: "...", assistant: "..."}
      session: telegram-123              # optional: access policy to apply
      sticky: [shopping]                 # optional: groups still active from
                                         # earlier turns (as after a tool ran)

To compare with code from before a skill got typed tools, LEGACY maps each
typed tool to the free-text ``skill_<name>`` call that did the same job; it is
used only when the expected tool is not offered (argument checks are skipped).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
LOCAL_CASES = Path.home() / ".eyetor/evals/tool_choice.yaml"
EXAMPLE_CASES = REPO / "evals/tool_choice.example.yaml"
DEFAULT_SESSION = "cli-eval"


LEGACY: dict[str, tuple[str, str]] = {
    "shopping_list_show": ("skill_shopping", r"^\s*list(\.py)?\s+show\b"),
    "shopping_list_add": ("skill_shopping", r"^\s*list(\.py)?\s+add\b"),
    "shopping_list_remove": ("skill_shopping", r"^\s*list(\.py)?\s+remove\b"),
    "shopping_list_clear": ("skill_shopping", r"^\s*list(\.py)?\s+clear\b"),
    "shopping_price_history": ("skill_shopping", r"^\s*price(\.py)?\s+history\b"),
    "shopping_price_cheapest": ("skill_shopping", r"^\s*price(\.py)?\s+cheapest\b"),
    "shopping_plan": ("skill_shopping", r"^\s*optimize(\.py)?\s+plan\b"),
    "web_search": ("skill_web_search", r"--query\b"),
    "web_fetch": ("skill_web_search", r"--fetch\b"),
    "desktop_screens_on": ("skill_hyprland_commands", r"^\s*hello\b"),
    "desktop_screens_off": ("skill_hyprland_commands", r"^\s*bye\b"),
    "desktop_lock": ("skill_hyprland_commands", r"^\s*lock\b"),
    "desktop_status": ("skill_hyprland_commands", r"^\s*status\b"),
    "desktop_screenshot": ("skill_hyprland_commands", r"^\s*shot\b"),
    "desktop_notify": ("skill_hyprland_commands", r"^\s*notify\b"),
    "desktop_volume": ("skill_hyprland_commands", r"^\s*volume\b"),
    "desktop_media": ("skill_hyprland_commands", r"^\s*media\b"),
}


def _legacy_match(expected: set[str], name: str, raw_args: str) -> bool:
    try:
        free_text = json.loads(raw_args or "{}").get("args", "")
    except (json.JSONDecodeError, AttributeError):
        return False
    return any(
        LEGACY[e][0] == name and re.search(LEGACY[e][1], free_text or "")
        for e in expected
        if e in LEGACY
    )


class _NoScheduler:
    """schedule_task & co. must be offered; they are never executed here."""


def _expected(case: dict) -> set[str]:
    exp = case.get("expect", "none")
    return {exp} if isinstance(exp, str) else set(exp)


def _args_match(want: dict, raw_args: str) -> bool:
    try:
        got = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return False
    for key, needle in (want or {}).items():
        value = got.get(key)
        hay = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if str(needle).lower() not in (hay or "").lower():
            return False
    return True


def judge(
    case: dict, tool_calls: list[tuple[str, str]], offered: set[str] | None = None
) -> tuple[bool, str]:
    """(passed, what the model did)."""
    expected = _expected(case)
    did = ", ".join(f"{n}({a[:80]})" for n, a in tool_calls) or "none"
    if expected == {"none"}:
        return not tool_calls, did
    if expected - {"none"} and offered is not None and not (expected & offered):
        legacy_ok = any(_legacy_match(expected, n, a) for n, a in tool_calls)
        return legacy_ok or ("none" in expected and not tool_calls), did
    ok = any(
        (n in expected and _args_match(case.get("args"), a)) for n, a in tool_calls
    ) or ("none" in expected and not tool_calls)
    return ok, did


async def run(cases: list[dict], provider: str | None, repeat: int) -> int:
    from eyetor.app import build_app
    from eyetor.cli import _resolve_provider
    from eyetor.config import load_config
    from eyetor.models.messages import Message

    cfg = load_config()
    cfg.sessions.persist = False
    prov = _resolve_provider(cfg, provider, None, None, None)
    app = await build_app(
        cfg, prov=prov, tracker=None, cost_estimator=None, use_host_tools=True,
        interactive=False, agent_provider=provider or "fallback", agent_model=prov.model,
    )
    manager = app.session_manager(scheduler=_NoScheduler())
    model = getattr(prov, "model", "?")
    print(f"provider={provider or 'fallback'} model={model} cases={len(cases)} repeat={repeat}\n")

    passed = total = 0
    failures: Counter[str] = Counter()
    t_start = time.monotonic()
    for i, case in enumerate(cases):
        wins = 0
        seen: list[str] = []
        for r in range(repeat):
            # A fresh session per attempt: gating state never leaks between cases.
            session = manager.get_or_create(f"{case.get('session', DEFAULT_SESSION)}#{i}.{r}")
            for turn in case.get("history", []):
                session._messages.append(Message(role="user", content=turn["user"]))
                session._messages.append(Message(role="assistant", content=turn["assistant"]))
            for group in case.get("sticky", []):
                session._sticky_groups[group] = session._sticky_turns + 1
            tool_defs, _ = session._turn_tool_defs(case["input"])
            offered = {t.name for t in tool_defs or []}
            messages = session._get_full_messages() + [Message(role="user", content=case["input"])]
            result = await prov.complete(
                messages=messages, tools=tool_defs, temperature=session.config.temperature
            )
            calls = [(tc.function.name, tc.function.arguments) for tc in result.message.tool_calls or []]
            ok, did = judge(case, calls, offered)
            wins += ok
            if not ok:
                seen.append(did if calls else f"none: {(result.message.content or '')[:80]!r}")
        passed += wins
        total += repeat
        mark = "✅" if wins == repeat else ("⚠️ " if wins else "❌")
        exp = "|".join(sorted(_expected(case)))
        print(f"{mark} {wins}/{repeat}  {case['input'][:60]!r} → {exp}")
        for s in dict.fromkeys(seen):
            print(f"        got: {s[:160]}")
        if wins < repeat:
            failures[exp] += repeat - wins

    elapsed = time.monotonic() - t_start
    print(f"\n{passed}/{total} = {100 * passed / max(total, 1):.0f}%  ({elapsed:.0f}s)")
    if failures:
        print("fallos por tool esperada:", ", ".join(f"{k}×{v}" for k, v in failures.most_common()))
    if app.mcp_registry is not None:
        await app.mcp_registry.close_all()
    return 0 if passed == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="?", type=Path)
    parser.add_argument("-p", "--provider", help="Provider name (default: fallback chain)")
    parser.add_argument("-n", "--repeat", type=int, default=1, help="Attempts per case")
    parser.add_argument("-k", "--filter", help="Only cases whose input or expect contains TEXT")
    args = parser.parse_args()

    path = args.cases or (LOCAL_CASES if LOCAL_CASES.exists() else EXAMPLE_CASES)
    cases = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if args.filter:
        needle = args.filter.lower()
        cases = [
            c for c in cases
            if needle in c["input"].lower() or needle in "|".join(_expected(c)).lower()
        ]
    print(f"cases: {path}")
    logging.basicConfig(level=logging.ERROR)
    return asyncio.run(run(cases, args.provider, args.repeat))


if __name__ == "__main__":
    sys.exit(main())
