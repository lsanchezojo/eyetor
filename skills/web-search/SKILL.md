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
2. The script returns JSON: `[{"title": "...", "url": "...", "snippet": "..."}]`
3. If the snippets contain enough information to fully answer the question, use them directly.
4. **If the snippets are too brief or don't contain the actual data needed** (weather, prices, scores, news body, etc.) you MUST fetch the most relevant URL before answering.

## How to fetch a URL
1. Run `scripts/search.py --fetch "<url>"`
2. Returns the page content as plain text (HTML stripped)
3. **Always fetch when the user needs specific data** that cannot be inferred from titles and snippets alone.

## When to always fetch (never answer from snippets alone)
- Weather / forecast queries → fetch a weather site URL from the results
- Sports scores or live results
- News article content (not just the headline)
- Prices, stock values, exchange rates
- Any question requiring up-to-date numeric or structured data

## Examples

Search then fetch weather:
```
scripts/search.py --query "tiempo Lebrija Sevilla hoy" --max-results 3
# pick best URL from results, then:
scripts/search.py --fetch "https://www.tiempo.es/lebrija.htm"
```

Search for an article:
```
scripts/search.py --query "OpenAI GPT-5 release" --max-results 3
# then fetch the most relevant article URL
scripts/search.py --fetch "https://..."
```

Search for reference docs:
```
scripts/search.py --query "Python asyncio tutorial" --max-results 3
# snippets are enough for general questions — no need to fetch
```

## Notes
- Search uses DuckDuckGo Lite (no API key required)
- Fetch strips HTML and returns readable text (truncated at 8000 chars)
- If a URL cannot be fetched, try the next result URL
- If a URL cannot be fetched, the script returns an error JSON object
