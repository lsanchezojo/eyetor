"""Hybrid retriever: BM25 + vector, fused with Reciprocal Rank Fusion (RRF).

Exposes a minimal ``SearchBackend`` protocol so the store can be swapped for
another backend (e.g. Qdrant) without touching callers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from eyetor.knowledge.embedder import Embedder
from eyetor.knowledge.store import ChunkRow, KnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    doc_id: int
    chunk_id: int
    workspace: str
    path: str
    title: str | None
    heading: str | None
    snippet: str
    score: float
    sources: list[str]


@dataclass
class ReadResult:
    doc_id: int
    path: str
    title: str | None
    section: str | None
    content: str
    truncated: bool
    total_chars: int


@runtime_checkable
class SearchBackend(Protocol):
    vector_enabled: bool

    def bm25_search(self, query: str, workspace: str | None, k: int) -> list[int]: ...
    def vector_search(
        self, query_embedding: list[float], workspace: str | None, k: int
    ) -> list[int]: ...
    def fetch_chunks(self, chunk_ids: list[int]) -> list[ChunkRow]: ...
    def snippet_for(self, chunk_id: int, query: str, max_tokens: int = 32) -> str: ...
    def read_chunks(
        self, doc_id: int, heading_prefix: str | None = None
    ) -> list[ChunkRow]: ...


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse multiple ranked lists of chunk ids into one via RRF."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


class Retriever:
    def __init__(
        self,
        backend: SearchBackend,
        embedder: Embedder | None = None,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidate_multiplier = max(1, candidate_multiplier)

    async def search(
        self,
        query: str,
        workspace: str | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        if not query.strip() or top_k <= 0:
            return []
        k_candidates = top_k * self.candidate_multiplier

        async def run_bm25() -> list[int]:
            return await asyncio.to_thread(
                self.backend.bm25_search, query, workspace, k_candidates
            )

        async def run_vector() -> list[int]:
            if self.embedder is None or not getattr(self.backend, "vector_enabled", False):
                return []
            try:
                vec = await asyncio.to_thread(self.embedder.embed_query, query)
            except Exception as exc:
                logger.warning("Query embedding failed: %s", exc)
                return []
            if not vec:
                return []
            return await asyncio.to_thread(
                self.backend.vector_search, vec, workspace, k_candidates
            )

        bm25_ids, vec_ids = await asyncio.gather(run_bm25(), run_vector())
        sources_map: dict[int, list[str]] = {}
        for cid in bm25_ids:
            sources_map.setdefault(cid, []).append("bm25")
        for cid in vec_ids:
            sources_map.setdefault(cid, []).append("vector")

        fused_ids = rrf_fuse([bm25_ids, vec_ids], k=self.rrf_k)[:top_k]
        if not fused_ids:
            return []

        rows = await asyncio.to_thread(self.backend.fetch_chunks, fused_ids)
        rank_score = {cid: 1.0 / (1 + i) for i, cid in enumerate(fused_ids)}
        hits: list[SearchHit] = []
        for row in rows:
            snippet = row.content[:400] + ("…" if len(row.content) > 400 else "")
            if "bm25" in sources_map.get(row.chunk_id, []):
                try:
                    snippet = await asyncio.to_thread(
                        self.backend.snippet_for, row.chunk_id, query, 32
                    )
                except Exception:
                    pass
            hits.append(
                SearchHit(
                    doc_id=row.doc_id,
                    chunk_id=row.chunk_id,
                    workspace=row.workspace,
                    path=row.rel_path,
                    title=row.title,
                    heading=row.heading_path,
                    snippet=snippet,
                    score=rank_score.get(row.chunk_id, 0.0),
                    sources=sources_map.get(row.chunk_id, []),
                )
            )
        return hits

    def read(
        self,
        doc_id: int,
        section_prefix: str | None = None,
        max_chars: int = 1800,
    ) -> ReadResult | None:
        rows = self.backend.read_chunks(doc_id, heading_prefix=section_prefix)
        if not rows:
            return None
        first = rows[0]
        parts: list[str] = []
        total = 0
        truncated = False
        for row in rows:
            piece = row.content
            if total + len(piece) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    parts.append(piece[:remaining])
                    total += remaining
                truncated = True
                break
            parts.append(piece)
            total += len(piece)
        return ReadResult(
            doc_id=doc_id,
            path=first.rel_path,
            title=first.title,
            section=section_prefix,
            content="\n\n".join(parts),
            truncated=truncated,
            total_chars=sum(len(r.content) for r in rows),
        )
