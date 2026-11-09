"""Knowledge-base tools: kb_search, kb_read, kb_list_sources."""

from __future__ import annotations

import json

from eyetor.models.tools import ToolDefinition, ToolRegistry


def register_kb_tools(tool_registry: ToolRegistry, knowledge) -> None:
    """Register the knowledge-base tools for a ``KnowledgeManager``."""
    _kb_ws_names = knowledge.list_workspaces()
    _kb_ws_desc = ", ".join(_kb_ws_names) if _kb_ws_names else "none"

    async def kb_search_handler(
        query: str, top_k: int = 5, workspace: str | None = None
    ) -> str:
        top_k = max(1, min(int(top_k or 5), 20))
        hits = await knowledge.search(query, workspace=workspace, top_k=top_k)
        return json.dumps(
            {
                "query": query,
                "count": len(hits),
                "results": [
                    {
                        "doc_id": h.doc_id,
                        "chunk_id": h.chunk_id,
                        "workspace": h.workspace,
                        "path": h.path,
                        "title": h.title,
                        "heading": h.heading,
                        "snippet": h.snippet,
                        "score": round(h.score, 4),
                        "sources": h.sources,
                    }
                    for h in hits
                ],
            },
            ensure_ascii=False,
        )

    async def kb_read_handler(
        doc_id: int, section: str | None = None, max_chars: int = 1800
    ) -> str:
        result = knowledge.read_doc(
            int(doc_id), section=section, max_chars=int(max_chars or 1800)
        )
        if result is None:
            return json.dumps(
                {"error": f"Document {doc_id} not found"}, ensure_ascii=False
            )
        if not result.section_matched:
            return json.dumps(
                {
                    "error": (
                        f"Section '{section}' not found in document {doc_id}."
                    ),
                    "doc_id": result.doc_id,
                    "path": result.path,
                    "title": result.title,
                    "available_sections": result.available_sections or [],
                    "hint": (
                        "Call kb_read without the 'section' parameter to "
                        "read the whole document, or pick one of "
                        "available_sections verbatim."
                    ),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "doc_id": result.doc_id,
                "path": result.path,
                "title": result.title,
                "section": result.section,
                "content": result.content,
                "truncated": result.truncated,
                "total_chars": result.total_chars,
            },
            ensure_ascii=False,
        )

    async def kb_list_sources_handler(
        workspace: str | None = None, limit: int = 50
    ) -> str:
        sources = knowledge.list_sources(
            workspace=workspace, limit=int(limit or 50)
        )
        return json.dumps(sources, ensure_ascii=False)

    tool_registry.register(
        ToolDefinition(
            name="kb_search",
            description=(
                "Search the workspace knowledge base for documentation, guides, "
                "specs and notes. Returns ranked snippets. Use this for conceptual "
                "or factual questions about project docs. For literal string matching "
                "in code, prefer the fs_grep tool. "
                f"Available workspaces: {_kb_ws_desc}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Query string. Supports FTS5 syntax: AND, OR, NEAR, "
                            '"quoted phrase".'
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1-20, default 5).",
                    },
                    "workspace": {
                        "type": "string",
                        "description": f"Optional workspace filter. One of: {_kb_ws_desc}.",
                    },
                },
                "required": ["query"],
            },
            handler=kb_search_handler,
            group="kb",
        )
    )

    tool_registry.register(
        ToolDefinition(
            name="kb_read",
            description=(
                "Read a specific document or section by id from the knowledge "
                "base. Use after kb_search to get more context around a match."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "integer",
                        "description": "Document id (from kb_search results).",
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional heading-path prefix to narrow the read.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 1800).",
                    },
                },
                "required": ["doc_id"],
            },
            handler=kb_read_handler,
            group="kb",
        )
    )

    tool_registry.register(
        ToolDefinition(
            name="kb_list_sources",
            description=(
                "List indexed documents in the knowledge base. Use to discover "
                "what documentation is available before searching."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": f"Optional workspace filter. One of: {_kb_ws_desc}.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents (default 50).",
                    },
                },
                "required": [],
            },
            handler=kb_list_sources_handler,
            group="kb",
        )
    )
