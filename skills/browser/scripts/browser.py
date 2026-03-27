#!/usr/bin/env python3
"""Browser control for the browser skill.

Usage:
    browser.py open       --url URL
    browser.py screenshot --url URL --output PATH
    browser.py source     --url URL
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import urllib.request


def _ok(data: dict) -> None:
    print(json.dumps({"ok": True, **data}, ensure_ascii=False))


def _err(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    sys.exit(1)


def cmd_open(url: str) -> None:
    """Open URL in the system default browser."""
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
        elif system == "Darwin":
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url])
        _ok({"opened": url, "browser": "system default"})
    except Exception as e:
        _err(str(e))


def cmd_screenshot(url: str, output: str) -> None:
    """Take a screenshot of a URL using playwright (if available)."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            page.screenshot(path=output, full_page=True)
            browser.close()
        _ok({"screenshot": output, "url": url})
    except ImportError:
        # Fallback: open in browser instead
        cmd_open(url)
        _err(
            "playwright not installed. Opened URL in browser instead. "
            "To enable screenshots: pip install playwright && playwright install chromium"
        )
    except Exception as e:
        _err(str(e))


def cmd_source(url: str) -> None:
    """Fetch and return page source as plain text."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Vector-Agent/1.0 (browser skill)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        _err(str(e))

    # Strip HTML tags for readability
    if "html" in content_type.lower():
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        raw = re.sub(r"<[^>]+>", "", raw)
        for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&quot;", '"')]:
            raw = raw.replace(entity, char)
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        raw = "\n".join(lines)
    if len(raw) > 10000:
        raw = raw[:10000] + "\n[truncated at 10000 chars]"
    _ok({"url": url, "content": raw, "content_type": content_type})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("open"); p.add_argument("--url", required=True)
    p = sub.add_parser("screenshot"); p.add_argument("--url", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("source"); p.add_argument("--url", required=True)

    args = parser.parse_args()
    if args.command == "open":         cmd_open(args.url)
    elif args.command == "screenshot": cmd_screenshot(args.url, args.output)
    elif args.command == "source":     cmd_source(args.url)


if __name__ == "__main__":
    main()
