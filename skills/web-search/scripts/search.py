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


def fetch(url: str, max_chars: int = 6000) -> str:
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

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[Content truncated at {max_chars} chars]"
    return text


def _decode_entities(text: str) -> str:
    """Decode common HTML entities in a plain-text string."""
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&quot;", '"'), ("&#x27;", "'")]:
        text = text.replace(entity, char)
    return text


def _html_to_text(html: str) -> str:
    """HTML → plain text with boilerplate removal.

    Drops scripts/styles/nav/header/footer/aside/form, prefers <main>/<article>
    when present, and de-duplicates consecutive identical lines (menu noise).
    """
    # Remove scripts, styles, templates and SVGs completely
    html = re.sub(
        r"<(script|style|template|svg|noscript)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove comments
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    # Drop boilerplate regions (nav, header, footer, aside, form)
    html = re.sub(
        r"<(nav|header|footer|aside|form)\b[^>]*>.*?</\1>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Prefer main content region if present
    main_match = re.search(
        r"<(main|article)\b[^>]*>(.*?)</\1>",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if main_match:
        html = main_match.group(2)
    # Replace common block tags with newlines
    html = re.sub(
        r"<(br|p|div|h[1-6]|li|tr|section|hr)[^>]*>",
        "\n",
        html,
        flags=re.IGNORECASE,
    )
    # Remove all remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Decode HTML entities
    html = _decode_entities(html)
    html = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), html)
    html = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), html)
    # Collapse whitespace per line
    lines = [re.sub(r"\s+", " ", line).strip() for line in html.splitlines()]
    lines = [l for l in lines if l]
    # De-duplicate consecutive identical lines (menu repetition)
    deduped: list[str] = []
    prev = None
    for l in lines:
        if l != prev:
            deduped.append(l)
            prev = l
    return "\n".join(deduped)


def _parse_ddg_lite(html: str) -> list[dict]:
    """Parse DuckDuckGo Lite HTML and extract results."""
    # DDG Lite uses single-quoted class attributes and redirect URLs
    link_pattern = re.compile(
        r"<a[^>]+class=['\"]result-link['\"][^>]*href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>"
        r"|<a[^>]+href=['\"]([^'\"]+)['\"][^>]+class=['\"]result-link['\"][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r"<td[^>]+class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
        re.IGNORECASE | re.DOTALL,
    )

    links = []
    for m in link_pattern.finditer(html):
        raw_url = m.group(1) or m.group(3)
        title = _decode_entities(re.sub(r"<[^>]+>", "", m.group(2) or m.group(4) or "").strip())
        # Decode redirect: //duckduckgo.com/l/?uddg=ENCODED_URL&rut=...
        uddg = re.search(r"[?&]uddg=([^&]+)", raw_url)
        url = urllib.parse.unquote(uddg.group(1)) if uddg else raw_url
        # Fix protocol-relative URLs
        if url.startswith("//"):
            url = "https:" + url
        links.append((url, title))

    snippets = [
        _decode_entities(re.sub(r"<[^>]+>", "", m.group(1)).strip())
        for m in snippet_pattern.finditer(html)
    ]

    results = []
    for i, (url, title) in enumerate(links):
        snippet = snippets[i] if i < len(snippets) else ""
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Web search and URL fetch")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--fetch", help="URL to fetch")
    parser.add_argument("--max-results", type=int, default=5, dest="max_results")
    parser.add_argument("--max-chars", type=int, default=6000, dest="max_chars",
                        help="Max chars returned by --fetch (default 6000)")
    args = parser.parse_args()

    if args.fetch:
        print(fetch(args.fetch, max_chars=args.max_chars))
    elif args.query:
        results = search(args.query, args.max_results)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
