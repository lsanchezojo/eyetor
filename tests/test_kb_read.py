"""Tests for kb_read section resolution and PDF page headings."""

from __future__ import annotations

from pathlib import Path

import pytest

from eyetor.knowledge.chunker import Chunker
from eyetor.knowledge.extractors import ExtractedDoc
from eyetor.knowledge.retriever import Retriever
from eyetor.knowledge.store import Chunk, KnowledgeStore


def _seed_doc(
    store: KnowledgeStore,
    *,
    workspace: str = "default",
    rel_path: str = "manual.md",
    title: str = "Manual",
    chunks: list[Chunk],
) -> int:
    store.upsert_workspace(workspace, "/tmp")
    doc_id = store.upsert_doc(
        workspace=workspace,
        rel_path=rel_path,
        abs_path=f"/tmp/{rel_path}",
        title=title,
        mtime=0.0,
        size_bytes=1,
        sha1="x" * 40,
        ext=".md",
    )
    store.replace_chunks(doc_id, chunks)
    return doc_id


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "kb.db")


# ── Store: read_chunks section filtering ──────────────────────────────


class TestReadChunksSection:
    def test_case_insensitive_match(self, store: KnowledgeStore) -> None:
        doc_id = _seed_doc(
            store,
            chunks=[
                Chunk(ordinal=0, heading_path="Líneas de habilidad", start_line=None, end_line=None, content="A"),
                Chunk(ordinal=1, heading_path="Otra sección", start_line=None, end_line=None, content="B"),
            ],
        )
        rows = store.read_chunks(doc_id, heading_prefix="líneas de habilidad")
        assert [r.content for r in rows] == ["A"]

    def test_substring_match(self, store: KnowledgeStore) -> None:
        doc_id = _seed_doc(
            store,
            chunks=[
                Chunk(ordinal=0, heading_path="Cap 3 > Líneas de habilidad", start_line=None, end_line=None, content="X"),
                Chunk(ordinal=1, heading_path="Cap 4 > Combate", start_line=None, end_line=None, content="Y"),
            ],
        )
        rows = store.read_chunks(doc_id, heading_prefix="Líneas")
        assert [r.content for r in rows] == ["X"]

    def test_empty_when_no_match(self, store: KnowledgeStore) -> None:
        doc_id = _seed_doc(
            store,
            chunks=[
                Chunk(ordinal=0, heading_path="Combate", start_line=None, end_line=None, content="Z"),
            ],
        )
        assert store.read_chunks(doc_id, heading_prefix="no-existe") == []


# ── Store: list_sections, doc_exists ──────────────────────────────────


class TestSectionsAndExists:
    def test_list_sections_order(self, store: KnowledgeStore) -> None:
        doc_id = _seed_doc(
            store,
            chunks=[
                Chunk(ordinal=0, heading_path="Intro", start_line=None, end_line=None, content="a"),
                Chunk(ordinal=1, heading_path="Intro", start_line=None, end_line=None, content="b"),
                Chunk(ordinal=2, heading_path="Combate", start_line=None, end_line=None, content="c"),
            ],
        )
        assert store.list_sections(doc_id) == ["Intro", "Combate"]

    def test_doc_exists(self, store: KnowledgeStore) -> None:
        doc_id = _seed_doc(
            store,
            chunks=[Chunk(ordinal=0, heading_path="X", start_line=None, end_line=None, content="x")],
        )
        assert store.doc_exists(doc_id) is True
        assert store.doc_exists(doc_id + 999) is False


# ── Retriever: doc-missing vs section-missing ─────────────────────────


class TestRetrieverRead:
    def test_missing_doc_returns_none(self, store: KnowledgeStore) -> None:
        retriever = Retriever(backend=store)
        assert retriever.read(doc_id=9999, section_prefix="anything") is None

    def test_section_missing_returns_actionable_result(self, store: KnowledgeStore) -> None:
        doc_id = _seed_doc(
            store,
            chunks=[
                Chunk(ordinal=0, heading_path="Intro", start_line=None, end_line=None, content="a"),
                Chunk(ordinal=1, heading_path="Combate", start_line=None, end_line=None, content="c"),
            ],
        )
        retriever = Retriever(backend=store)
        result = retriever.read(doc_id=doc_id, section_prefix="no-existe")
        assert result is not None
        assert result.section_matched is False
        assert result.content == ""
        assert result.available_sections == ["Intro", "Combate"]

    def test_section_matches_case_insensitive(self, store: KnowledgeStore) -> None:
        doc_id = _seed_doc(
            store,
            chunks=[
                Chunk(ordinal=0, heading_path="Líneas de habilidad", start_line=None, end_line=None, content="contenido"),
            ],
        )
        retriever = Retriever(backend=store)
        # COLLATE NOCASE handles ASCII case only; diacritics must match.
        result = retriever.read(doc_id=doc_id, section_prefix="líneas DE habilidad")
        assert result is not None
        assert result.section_matched is True
        assert "contenido" in result.content


# ── Chunker: PDF page markers become heading_path ─────────────────────


class TestChunkerPageMarkers:
    def test_page_markers_become_heading(self) -> None:
        text = (
            "\n\n[Page 1]\n\nIntro contenido inicial.\n\n"
            "[Page 2]\n\nSegunda página contenido.\n\n"
            "[Page 3]\n\nTercera página."
        )
        doc = ExtractedDoc(text=text, title="Manual", sections=[])
        chunks = Chunker(max_chars=1500, overlap_chars=0).chunk(doc, suffix=".pdf")
        assert chunks, "chunker must produce at least one chunk"
        # Every chunk should have a Page N heading (not the title fallback)
        assert all(
            c.heading_path and c.heading_path.startswith("Page ") for c in chunks
        )

    def test_chunk_without_marker_inherits_previous_page(self) -> None:
        # Single chunk with multiple page markers — first one wins.
        text = "[Page 5]\n\nBloque con marcador.\n\nContinúa sin marcador."
        doc = ExtractedDoc(text=text, title="Manual", sections=[])
        chunks = Chunker(max_chars=1500, overlap_chars=0).chunk(doc, suffix=".pdf")
        assert chunks[0].heading_path == "Page 5"
