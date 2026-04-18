"""High-level KnowledgeManager wiring store, embedder, indexer and retriever."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from eyetor.knowledge.embedder import Embedder
from eyetor.knowledge.indexer import IndexReport, Indexer, WorkspaceSpec
from eyetor.knowledge.retriever import ReadResult, Retriever, SearchHit
from eyetor.knowledge.store import KnowledgeStore

if TYPE_CHECKING:
    from eyetor.config import KnowledgeConfig

logger = logging.getLogger(__name__)

_CWD_WORKSPACE = "cwd"


class KnowledgeManager:
    def __init__(
        self,
        store: KnowledgeStore,
        indexer: Indexer,
        retriever: Retriever,
        workspaces: dict[str, WorkspaceSpec],
        *,
        top_k_default: int = 5,
        snippet_chars: int = 400,
    ) -> None:
        self.store = store
        self.indexer = indexer
        self.retriever = retriever
        self.workspaces = workspaces
        self.top_k_default = top_k_default
        self.snippet_chars = snippet_chars

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: "KnowledgeConfig") -> "KnowledgeManager":
        embedder = Embedder.from_config(cfg.embedding)
        dim = embedder.dim if embedder is not None else 384
        store = KnowledgeStore.from_path(cfg.db_path, vector_dim=dim)
        indexer = Indexer(
            store=store,
            embedder=embedder,
            chunk_max_chars=cfg.chunk.max_chars,
            chunk_overlap_chars=cfg.chunk.overlap_chars,
            max_file_size_bytes=cfg.max_file_size_bytes,
        )
        retriever = Retriever(
            backend=store,
            embedder=embedder,
            rrf_k=cfg.retrieval.rrf_k,
            candidate_multiplier=cfg.retrieval.candidate_multiplier,
        )
        workspaces: dict[str, WorkspaceSpec] = {}
        for ws in cfg.workspaces:
            workspaces[ws.name] = WorkspaceSpec(
                name=ws.name,
                root=Path(ws.path).expanduser(),
                include=list(ws.include),
                exclude=list(ws.exclude),
            )
            store.upsert_workspace(ws.name, str(Path(ws.path).expanduser()))
        return cls(
            store=store,
            indexer=indexer,
            retriever=retriever,
            workspaces=workspaces,
            top_k_default=cfg.retrieval.top_k_default,
            snippet_chars=cfg.retrieval.snippet_chars,
        )

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    def register_cwd_workspace(self, path: Path) -> None:
        path = path.expanduser().resolve()
        self.workspaces[_CWD_WORKSPACE] = WorkspaceSpec(
            name=_CWD_WORKSPACE,
            root=path,
            include=[],  # uses extractor default globs
            exclude=[
                "**/node_modules/**",
                "**/.git/**",
                "**/.venv/**",
                "**/venv/**",
                "**/__pycache__/**",
                "**/build/**",
                "**/dist/**",
                "**/.next/**",
                "**/target/**",
            ],
        )
        self.store.upsert_workspace(_CWD_WORKSPACE, str(path))
        logger.info("kb: registered cwd workspace at %s", path)

    def list_workspaces(self) -> list[str]:
        return sorted(self.workspaces.keys())

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_workspace(
        self, name: str, *, force: bool = False, prune: bool = True
    ) -> IndexReport:
        spec = self.workspaces.get(name)
        if not spec:
            raise KeyError(f"Unknown workspace: {name}")
        return await self.indexer.index_workspace(spec, force=force, prune=prune)

    async def index_all(
        self, *, force: bool = False, prune: bool = True
    ) -> dict[str, IndexReport]:
        if not self.workspaces:
            return {}
        tasks = {
            name: asyncio.create_task(
                self.indexer.index_workspace(spec, force=force, prune=prune)
            )
            for name, spec in self.workspaces.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out: dict[str, IndexReport] = {}
        for name, res in zip(tasks.keys(), results):
            if isinstance(res, BaseException):
                logger.warning("kb: workspace %s failed: %s", name, res)
                continue
            out[name] = res
        return out

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def search(
        self, query: str, workspace: str | None = None, top_k: int | None = None
    ) -> list[SearchHit]:
        return await self.retriever.search(
            query=query,
            workspace=workspace,
            top_k=top_k or self.top_k_default,
        )

    def read_doc(
        self, doc_id: int, section: str | None = None, max_chars: int = 1800
    ) -> ReadResult | None:
        return self.retriever.read(doc_id, section_prefix=section, max_chars=max_chars)

    def list_sections(self, doc_id: int, limit: int = 20) -> list[str]:
        return self.store.list_sections(doc_id, limit=limit)

    def list_sources(
        self, workspace: str | None = None, limit: int = 50
    ) -> dict:
        docs = self.store.list_docs(workspace=workspace, limit=limit)
        total = len(docs)
        return {
            "workspaces": [ws["name"] for ws in self.store.list_workspaces()],
            "docs": [
                {
                    "doc_id": d.id,
                    "workspace": d.workspace,
                    "path": d.rel_path,
                    "title": d.title,
                }
                for d in docs
            ],
            "total": total,
        }

    def stats(self) -> dict:
        return self.store.stats()

    # ------------------------------------------------------------------
    # System-prompt injection
    # ------------------------------------------------------------------

    def build_context(self) -> str:
        stats = self.store.stats()
        ws_rows = self.store.list_workspaces()
        if not ws_rows:
            return ""
        summaries: list[str] = []
        for ws in ws_rows:
            docs = self.store.list_docs(workspace=ws["name"], limit=1000)
            summaries.append(f"{ws['name']} ({len(docs)} docs)")
        vector_note = (
            " (hybrid BM25 + semantic)"
            if stats.get("vector_enabled")
            else " (BM25 only — install eyetor[knowledge-vector] for semantic)"
        )
        lines = [
            "## Knowledge Base (on-demand)" + vector_note,
            "Workspaces available: " + ", ".join(summaries) + ".",
            "Use kb_search(query) to retrieve ranked snippets from documentation, guides, specs and notes.",
            "Use kb_read(doc_id, section) for more context around a match.",
            "Use kb_list_sources() to discover available documents.",
            "Prefer kb_search for conceptual or factual questions; use the filesystem grep skill for literal strings.",
        ]
        return "\n".join(lines)
