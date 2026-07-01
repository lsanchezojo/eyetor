"""Tests for knowledge-base document expiry (caducidad) and single-file ingest.

Covers the ephemeral-upload path used by the Telegram document handler:
- ``expires_at`` migration/column on the ``docs`` table
- ``KnowledgeStore.purge_expired`` removing only past-due docs (and their chunks)
- ``Indexer.index_file`` / ``KnowledgeManager.ingest_upload`` round-trips
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from eyetor.knowledge.indexer import Indexer
from eyetor.knowledge.manager import KnowledgeManager
from eyetor.knowledge.retriever import Retriever
from eyetor.knowledge.store import Chunk, KnowledgeStore


def _iso(days: float) -> str:
    return (datetime.utcnow() + timedelta(days=days)).isoformat()


def _seed(store: KnowledgeStore, rel_path: str, *, expires_at: str | None) -> int:
    store.upsert_workspace("up", "/tmp")
    doc_id = store.upsert_doc(
        workspace="up",
        rel_path=rel_path,
        abs_path=f"/tmp/{rel_path}",
        title=rel_path,
        mtime=0.0,
        size_bytes=1,
        sha1="a" * 40,
        ext=".txt",
        expires_at=expires_at,
    )
    store.replace_chunks(
        doc_id,
        [Chunk(ordinal=0, heading_path=None, start_line=None, end_line=None, content="hola")],
    )
    return doc_id


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "kb.db")


class TestPurgeExpired:
    def test_column_present_after_migration(self, store: KnowledgeStore) -> None:
        cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(docs)")}
        assert "expires_at" in cols

    def test_purges_only_past_due(self, store: KnowledgeStore) -> None:
        past = _seed(store, "past.txt", expires_at=_iso(-1))
        future = _seed(store, "future.txt", expires_at=_iso(+1))
        never = _seed(store, "never.txt", expires_at=None)

        purged = store.purge_expired()

        assert purged == 1
        assert store.doc_exists(past) is False
        assert store.doc_exists(future) is True
        assert store.doc_exists(never) is True

    def test_purge_removes_chunks(self, store: KnowledgeStore) -> None:
        doc_id = _seed(store, "past.txt", expires_at=_iso(-1))
        assert store.read_chunks(doc_id)  # present before purge
        store.purge_expired()
        assert store.read_chunks(doc_id) == []

    def test_explicit_now_boundary(self, store: KnowledgeStore) -> None:
        # expires_at exactly "now+1h" is not purged when now is the reference.
        doc_id = _seed(store, "soon.txt", expires_at=_iso(+0.04))
        assert store.purge_expired(now_iso=datetime.utcnow().isoformat()) == 0
        assert store.doc_exists(doc_id) is True


def _manager(store: KnowledgeStore) -> KnowledgeManager:
    # BM25-only (embedder=None) to avoid loading fastembed in tests.
    indexer = Indexer(store=store, embedder=None)
    retriever = Retriever(backend=store)
    return KnowledgeManager(store=store, indexer=indexer, retriever=retriever, workspaces={})


class TestIngestUpload:
    def test_roundtrip_and_scoped_search(self, store: KnowledgeStore, tmp_path: Path) -> None:
        mgr = _manager(store)
        doc = tmp_path / "notas.txt"
        doc.write_text("El presupuesto trimestral asciende a doce mil euros.", encoding="utf-8")

        result = asyncio.run(mgr.ingest_upload(4242, doc, retention_days=7))

        assert result is not None
        assert result["workspace"] == "tg-upload-4242"
        assert result["expires_at"] is not None

        hits = asyncio.run(mgr.search("presupuesto trimestral", workspace="tg-upload-4242"))
        assert hits, "ingested upload must be retrievable in its own workspace"

    def test_no_retention_means_no_expiry(self, store: KnowledgeStore, tmp_path: Path) -> None:
        mgr = _manager(store)
        doc = tmp_path / "sin_caducidad.txt"
        doc.write_text("contenido perdurable", encoding="utf-8")

        result = asyncio.run(mgr.ingest_upload(1, doc, retention_days=0))

        assert result is not None
        assert result["expires_at"] is None
        assert mgr.purge_expired() == 0

    def test_fractional_retention_days(self, store: KnowledgeStore, tmp_path: Path) -> None:
        mgr = _manager(store)
        doc = tmp_path / "una_hora.txt"
        doc.write_text("caduca en una hora", encoding="utf-8")

        # ~1 hour of retention (0.042 days) → expires within the next 2 hours.
        result = asyncio.run(mgr.ingest_upload(1, doc, retention_days=0.042))

        assert result is not None
        expires = datetime.fromisoformat(result["expires_at"])
        assert datetime.utcnow() < expires < datetime.utcnow() + timedelta(hours=2)

    def test_unsupported_file_returns_none(self, store: KnowledgeStore, tmp_path: Path) -> None:
        mgr = _manager(store)
        blob = tmp_path / "archivo.zip"
        blob.write_bytes(b"PK\x03\x04binary")

        assert asyncio.run(mgr.ingest_upload(1, blob, retention_days=7)) is None

    def test_ingested_then_expired_is_purged(self, store: KnowledgeStore, tmp_path: Path) -> None:
        mgr = _manager(store)
        doc = tmp_path / "efimero.txt"
        doc.write_text("dato temporal que caduca", encoding="utf-8")

        result = asyncio.run(mgr.ingest_upload(7, doc, retention_days=7))
        assert result is not None
        # Force the doc into the past, then purge.
        store._conn.execute(
            "UPDATE docs SET expires_at = ? WHERE id = ?", (_iso(-1), result["doc_id"])
        )
        store._conn.commit()

        assert mgr.purge_expired() == 1
        hits = asyncio.run(mgr.search("dato temporal", workspace="tg-upload-7"))
        assert hits == []
