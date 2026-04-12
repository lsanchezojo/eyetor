"""Workspace indexer: walks a directory and keeps the KB in sync with disk.

Incremental by sha1 over **raw bytes** — binary formats (pdf/docx/xlsx/pptx)
are covered without having to re-extract them to compare text. The pipeline is
extractor → chunker → embedder (optional) → store.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from eyetor.knowledge.chunker import Chunker
from eyetor.knowledge.embedder import Embedder
from eyetor.knowledge.extractors import get_extractor, supported_extensions
from eyetor.knowledge.store import Chunk, KnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceSpec:
    name: str
    root: Path
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class IndexReport:
    workspace: str
    root: str
    scanned: int = 0
    indexed: int = 0
    updated: int = 0
    skipped: int = 0
    pruned: int = 0
    errors: int = 0
    chunks_written: int = 0
    duration_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "workspace": self.workspace,
            "root": self.root,
            "scanned": self.scanned,
            "indexed": self.indexed,
            "updated": self.updated,
            "skipped": self.skipped,
            "pruned": self.pruned,
            "errors": self.errors,
            "chunks_written": self.chunks_written,
            "duration_s": round(self.duration_s, 2),
        }


class Indexer:
    def __init__(
        self,
        store: KnowledgeStore,
        embedder: Embedder | None = None,
        *,
        chunk_max_chars: int = 1500,
        chunk_overlap_chars: int = 150,
        max_file_size_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunker = Chunker(max_chars=chunk_max_chars, overlap_chars=chunk_overlap_chars)
        self.max_file_size_bytes = max_file_size_bytes

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def index_workspace(
        self, spec: WorkspaceSpec, *, force: bool = False, prune: bool = True
    ) -> IndexReport:
        return await asyncio.to_thread(
            self._index_sync, spec, force=force, prune=prune
        )

    def _index_sync(
        self, spec: WorkspaceSpec, *, force: bool, prune: bool
    ) -> IndexReport:
        start = time.monotonic()
        report = IndexReport(workspace=spec.name, root=str(spec.root))
        root = spec.root.expanduser().resolve()
        if not root.exists():
            logger.warning("Workspace root does not exist: %s", root)
            report.duration_s = time.monotonic() - start
            return report

        self.store.upsert_workspace(spec.name, str(root))

        includes = spec.include or [f"**/*{ext}" for ext in supported_extensions()]
        excludes = spec.exclude or []
        seen_rel: set[str] = set()

        candidates = self._walk(root, includes, excludes)
        for abs_path in candidates:
            report.scanned += 1
            try:
                rel = str(abs_path.relative_to(root))
            except ValueError:
                continue
            seen_rel.add(rel)
            try:
                if abs_path.stat().st_size > self.max_file_size_bytes:
                    report.skipped += 1
                    continue
            except OSError:
                report.errors += 1
                continue

            try:
                raw = abs_path.read_bytes()
            except OSError as exc:
                logger.debug("Read failed %s: %s", abs_path, exc)
                report.errors += 1
                continue

            sha1 = hashlib.sha1(raw).hexdigest()
            existing = self.store.get_doc(spec.name, rel)
            if not force and existing and existing.sha1 == sha1:
                report.skipped += 1
                continue

            extractor = get_extractor(abs_path.suffix)
            if extractor is None:
                report.skipped += 1
                continue

            try:
                extracted = extractor(abs_path)
            except Exception as exc:
                logger.warning("Extractor failed for %s: %s", abs_path, exc)
                report.errors += 1
                continue

            if not extracted or not extracted.text.strip():
                report.skipped += 1
                continue

            chunks = self.chunker.chunk(extracted, suffix=abs_path.suffix)
            if not chunks:
                report.skipped += 1
                continue

            embeddings = self._embed_chunks(chunks)

            try:
                stat = abs_path.stat()
                doc_id = self.store.upsert_doc(
                    workspace=spec.name,
                    rel_path=rel,
                    abs_path=str(abs_path),
                    title=extracted.title,
                    mtime=stat.st_mtime,
                    size_bytes=stat.st_size,
                    sha1=sha1,
                    ext=abs_path.suffix.lower(),
                )
                self.store.replace_chunks(doc_id, chunks, embeddings)
            except Exception as exc:
                logger.warning("Store write failed for %s: %s", abs_path, exc)
                report.errors += 1
                continue

            if existing:
                report.updated += 1
            else:
                report.indexed += 1
            report.chunks_written += len(chunks)

        if prune:
            report.pruned = self._prune_removed(spec.name, seen_rel)

        report.duration_s = time.monotonic() - start
        logger.info(
            "kb: workspace=%s scanned=%d indexed=%d updated=%d skipped=%d pruned=%d errors=%d chunks=%d in %.2fs",
            report.workspace,
            report.scanned,
            report.indexed,
            report.updated,
            report.skipped,
            report.pruned,
            report.errors,
            report.chunks_written,
            report.duration_s,
        )
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _walk(
        self, root: Path, includes: list[str], excludes: list[str]
    ) -> list[Path]:
        results: list[Path] = []
        seen: set[Path] = set()
        for pattern in includes:
            for p in root.glob(pattern):
                if not p.is_file():
                    continue
                try:
                    rel = p.relative_to(root).as_posix()
                except ValueError:
                    continue
                if any(fnmatch.fnmatch(rel, ex) for ex in excludes):
                    continue
                resolved = p.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                results.append(p)
        return results

    def _embed_chunks(self, chunks: list[Chunk]) -> list[list[float]] | None:
        if self.embedder is None or not chunks:
            return None
        try:
            return self.embedder.embed_documents([c.content for c in chunks])
        except Exception as exc:
            logger.warning("Embedding failed, falling back to BM25-only for this doc: %s", exc)
            return None

    def _prune_removed(self, workspace: str, seen_rel: set[str]) -> int:
        pruned = 0
        for doc_id, rel in self.store.all_doc_rel_paths(workspace):
            if rel not in seen_rel:
                self.store.delete_doc(doc_id)
                pruned += 1
        return pruned
