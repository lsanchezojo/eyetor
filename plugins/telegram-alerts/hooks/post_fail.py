#!/usr/bin/env python3
"""Post-tool-use-failure hook: alert on tool errors via Telegram."""

import json
import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        raw = json.load(f)
    for key, val in raw.items():
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            raw[key] = os.environ.get(val[2:-1], "")
    return raw


def send_telegram(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}", file=sys.stderr)


def main() -> None:
    config = load_config()

    if not config.get("alert_on_failure"):
        return
    if not config.get("bot_token") or not config.get("chat_id"):
        return

    tool_name = os.environ.get("HOOK_TOOL_NAME", "")
    tool_error = os.environ.get("HOOK_TOOL_ERROR", "unknown error")
    duration_str = os.environ.get("HOOK_TOOL_DURATION_MS", "")

    if tool_name in config.get("ignore_tools", []):
        return

    duration_info = ""
    if duration_str:
        secs = int(duration_str) / 1000
        duration_info = f" (tras {secs:.1f}s)"

    # Truncar error si es muy largo
    if len(tool_error) > 300:
        tool_error = tool_error[:300] + "..."

    text = (
        f"\u274c <b>Tool fallida</b>{duration_info}\n"
        f"<code>{tool_name}</code>\n"
        f"<pre>{tool_error}</pre>"
    )
    send_telegram(config["bot_token"], config["chat_id"], text)


if __name__ == "__main__":
    main()
