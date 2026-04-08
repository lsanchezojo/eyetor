#!/usr/bin/env python3
"""Post-tool-use hook: alert on slow tool executions via Telegram."""

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
    # Resolve ${ENV_VAR} patterns
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

    if not config.get("alert_on_slow"):
        return
    if not config.get("bot_token") or not config.get("chat_id"):
        return

    tool_name = os.environ.get("HOOK_TOOL_NAME", "")
    duration_str = os.environ.get("HOOK_TOOL_DURATION_MS", "")

    if tool_name in config.get("ignore_tools", []):
        return
    if not duration_str:
        return

    duration_ms = int(duration_str)
    threshold = config.get("slow_threshold_ms", 10000)

    if duration_ms >= threshold:
        secs = duration_ms / 1000
        text = (
            f"\u26a0\ufe0f <b>Tool lenta</b>\n"
            f"<code>{tool_name}</code> tard\u00f3 <b>{secs:.1f}s</b> "
            f"(umbral: {threshold / 1000:.0f}s)"
        )
        send_telegram(config["bot_token"], config["chat_id"], text)


if __name__ == "__main__":
    main()
