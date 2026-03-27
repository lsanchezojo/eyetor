---
name: browser
description: Open URLs in the system default browser, take screenshots of web pages, and capture page content. Use when the user wants to open a website, preview a URL, or capture a screenshot.
license: MIT
compatibility: Python 3.11+. Works on Windows (uses start), Linux (xdg-open), macOS (open).
metadata:
  author: eyetor
  version: "1.0"
---

# Browser

## When to use this skill
Use when the user asks to:
- Open a website or URL in their browser
- Preview a web page
- Take a screenshot of a URL
- Open a local HTML file in the browser

## How to open a URL
```
scripts/browser.py open --url "https://example.com"
```
Opens the URL in the system default browser.

## How to open a local file
```
scripts/browser.py open --url "file:///C:/Users/user/report.html"
```

## How to take a screenshot (requires playwright or selenium)
```
scripts/browser.py screenshot --url "https://example.com" --output "/tmp/screenshot.png"
```
Falls back gracefully if playwright is not installed.

## How to fetch page source (no browser, just HTTP)
```
scripts/browser.py source --url "https://example.com"
```

## Notes
- `open` works without any additional dependencies
- `screenshot` requires `playwright` (optional): `pip install playwright && playwright install chromium`
- `source` uses httpx (already a project dependency)
- Returns JSON with status and path/content
