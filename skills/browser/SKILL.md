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

## Usage

Call `run_skill_script` with `skill="browser"`, `script="browser.py"`, and `args` set to one of the commands below.

### Fetch page content (most common — use this to read a URL)
- args: `source --url "https://example.com"`
- If you omit the subcommand, `source` is assumed: `--url "https://example.com"` also works.

### Open URL in system browser
- args: `open --url "https://example.com"`

### Take a screenshot
- args: `screenshot --url "https://example.com" --output "/tmp/screenshot.png"`
- Requires playwright. Falls back gracefully if not installed.

## Notes
- `source` strips HTML and returns plain text (truncated at 10k chars)
- Returns JSON: `{"ok": true, ...}` on success, `{"ok": false, "error": "..."}` on failure
