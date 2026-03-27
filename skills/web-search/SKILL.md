---
name: web-search
description: Search the web for information and fetch URL content. Use when the user needs to find information online, retrieve web page content, or get up-to-date information.
license: MIT
compatibility: Requires Python 3.11+ and httpx
metadata:
  author: eyetor
  version: "1.0"
---

# Web Search

## When to use this skill
Use this skill when:
- The user asks about current events or news
- The user needs to find specific information online
- The user provides a URL to fetch content from
- You need to verify or look up facts

## How to search the web
1. Run `scripts/search.py --query "<search terms>" --max-results 5`
2. The script returns JSON with a list of results: `[{"title": "...", "url": "...", "snippet": "..."}]`
3. Pick the most relevant results to answer the user's question
4. Optionally fetch the full content of a result URL (see below)

## How to fetch a URL
1. Run `scripts/search.py --fetch "<url>"`
2. Returns the page content as plain text (HTML stripped)
3. Use this when a search result snippet is not enough

## Examples

Search for Python asyncio:
```
scripts/search.py --query "Python asyncio tutorial" --max-results 3
```

Fetch a specific page:
```
scripts/search.py --fetch "https://docs.python.org/3/library/asyncio.html"
```

## Notes
- Search uses DuckDuckGo Lite (no API key required)
- Fetch strips HTML and returns readable text
- Respect robots.txt and rate limits
- If a URL cannot be fetched, the script returns an error JSON object
