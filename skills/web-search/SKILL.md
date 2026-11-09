---
name: web-search
description: Search the web for information and fetch URL content. Use when the user needs to find information online, retrieve web page content, or get up-to-date information.
license: MIT
compatibility: Requires Python 3.11+ and httpx
metadata:
  author: eyetor
  version: "2.0"
# Only typed tools: with free-text args small models dropped --query.
args_tool: false
tools:
  - name: web_search
    description: Search the web. Returns title, url and snippet per result.
    script: search.py
    parameters:
      type: object
      properties:
        query: {type: string, description: All search terms in one string.}
        max_results: {type: integer, description: "Number of results (default 5)."}
      required: [query]
  - name: web_fetch
    description: Fetch a web page as plain text (HTML stripped).
    script: search.py
    parameters:
      type: object
      properties:
        url: {type: string, description: Page URL.}
        max_chars: {type: integer, description: "Maximum characters returned (default 6000); use ~3000 when you only need one fact."}
      required: [url]
---

# Web Search

## When to use this skill
Use this skill when:
- The user asks about current events or news
- The user needs to find specific information online
- The user provides a URL to fetch content from
- You need to verify or look up facts

## How to search the web
1. Call `web_search` with all the search terms in `query`.
2. If the snippets contain enough information to fully answer the question, use them directly.
3. **If the snippets are too brief or don't contain the actual data needed** (weather, prices, scores, news body, etc.) call `web_fetch` on the most relevant URL.

## When to always fetch (never answer from snippets alone)
- Weather / forecast queries → fetch a weather site URL from the results
- Sports scores or live results
- News article content (not just the headline)
- Prices, stock values, exchange rates
- Any question requiring up-to-date numeric or structured data

## Estrategia de investigación web
1. **Planifica antes de buscar**: identifica las 1-3 preguntas concretas que necesitas resolver antes de lanzar ninguna query. No improvises búsquedas en cadena.
2. **Calidad sobre cantidad**: máximo **3-4 queries** de búsqueda y **2-3 fetches** por consulta del usuario. Si con eso no puedes responder, sintetiza lo que tengas y di honestamente qué falta.
3. **Elige fuentes primarias**: prefiere documentación oficial, páginas de producto del fabricante, papers o HuggingFace model cards. Evita agregadores y blogs secundarios salvo que sean la única fuente.
4. **No persigas todos los ángulos**: si el usuario pregunta por un modelo, dile que puede haber otros ángulos, pero céntrate en lo que pide.
5. **Sintetiza pronto**: en cuanto tengas información suficiente para dar una respuesta útil, **para de buscar** y responde. Siempre es mejor una respuesta con 3 datos confirmados que seguir buscando el cuarto.
6. **Cita fuentes**: incluye las URLs consultadas al final de la respuesta para que el usuario pueda verificar.
7. **Controla el tamaño de fetch**: si una página es larga y sólo necesitas un dato, usa `--max-chars 3000` para limitar el output.

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
