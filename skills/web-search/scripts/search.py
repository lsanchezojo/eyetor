#!/usr/bin/env python3
"""Web search and URL fetch script for the web-search skill.

Usage:
    search.py --query "search terms" [--max-results N]
    search.py --fetch "https://example.com"

Output: JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request


def search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo Lite and return result list."""
    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Vector-Agent/1.0 (web-search skill)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return [{"error": str(e)}]

    results = _parse_ddg_lite(html)
    return results[:max_results]


def fetch(url: str) -> str:
    """Fetch a URL and return readable plain text."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Vector-Agent/1.0 (web-search skill)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return json.dumps({"error": str(e)})

    if "html" in content_type.lower():
        text = _html_to_text(raw)
    else:
        text = raw

    # Truncate to 8000 chars to avoid context overflow
    if len(text) > 8000:
        text = text[:8000] + "\n\n[Content truncated at 8000 chars]"
    return text


def _html_to_text(html: str) -> str:
    """Very basic HTML to plain text conversion."""
    # Remove scripts and styles
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace common block tags with newlines
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Remove all remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Decode common HTML entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&quot;", '"')]:
        html = html.replace(entity, char)
    # Collapse whitespace
    lines = [line.strip() for line in html.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


def _parse_ddg_lite(html: str) -> list[dict]:
    """Parse DuckDuckGo Lite HTML and extract results."""
    results = []
    # Find result links — DDG Lite uses <a class="result-link">
    link_pattern = re.compile(
        r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>',
        re.IGNORECASE | re.DOTALL,
    )
    links = link_pattern.findall(html)
    snippets = [m for m in snippet_pattern.findall(html)]

    for i, (url, title) in enumerate(links):
        title = re.sub(r"<[^>]+>", "", title).strip()
        snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Web search and URL fetch")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--fetch", help="URL to fetch")
    parser.add_argument("--max-results", type=int, default=5, dest="max_results")
    args = parser.parse_args()

    if args.fetch:
        print(fetch(args.fetch))
    elif args.query:
        results = search(args.query, args.max_results)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
