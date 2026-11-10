"""Vector side of the knowledge base: backfill, workspace filter, locking.

Uses sqlite-vec with a fake embedder, so no model is downloaded.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from eyetor.knowledge.indexer import Indexer, WorkspaceSpec  # noqa: E402
from eyetor.knowledge.store import KnowledgeStore  # noqa: E402

DIM = 8


class FakeEmbedder:
    """Deterministic bag-of-letters vectors; counts calls."""

    dim = DIM

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for ch in text.lower():
            if ch.isalpha():
                v[ord(ch) % DIM] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _workspace(tmp_path: Path, name: str, files: dict[str, str]) -> WorkspaceSpec:
    root = tmp_path / name
    root.mkdir()
    for rel, text in files.items():
        (root / rel).write_text(text, encoding="utf-8")
    return WorkspaceSpec(name=name, root=root, include=["*.md"], exclude=[])


def _indexer(store: KnowledgeStore, embedder) -> Indexer:
    return Indexer(store, embedder)


def _vec_count(store: KnowledgeStore) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]


def test_unchanged_docs_indexed_without_embedder_get_backfilled(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db", vector_dim=DIM)
    assert store.vector_enabled
    spec = _workspace(tmp_path, "notes", {"a.md": "# Gatos\nlos gatos duermen", "b.md": "# Perros\nladran"})

    # Indexed while fastembed was missing: chunks but no vectors.
    first = _indexer(store, None)._index_sync(spec, force=False, prune=True)
    assert first.indexed == 2 and _vec_count(store) == 0

    # Same files, embedder now available: docs are skipped (same sha1) but
    # their chunks get vectors.
    emb = FakeEmbedder()
    second = _indexer(store, emb)._index_sync(spec, force=False, prune=True)
    chunks = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert second.skipped == 2
    assert second.vectors_backfilled == chunks == _vec_count(store)

    # Nothing left to do on the next run.
    third = _indexer(store, emb)._index_sync(spec, force=False, prune=True)
    assert third.vectors_backfilled == 0


def test_backfill_stops_on_wrong_dimension(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db", vector_dim=DIM)
    spec = _workspace(tmp_path, "notes", {"a.md": "# A\ntexto"})
    _indexer(store, None)._index_sync(spec, force=False, prune=True)

    class WrongDim(FakeEmbedder):
        def embed_documents(self, texts):
            self.calls += 1
            return [[1.0] * (DIM + 1) for _ in texts]

    emb = WrongDim()
    report = _indexer(store, emb)._index_sync(spec, force=False, prune=True)
    assert report.vectors_backfilled == 0 and emb.calls == 1  # no infinite loop


def test_vector_search_finds_small_workspace_next_to_big_one(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db", vector_dim=DIM)
    emb = FakeEmbedder()
    big = _workspace(tmp_path, "big", {f"f{i}.md": f"# T{i}\nzzzz qqqq {i}" for i in range(60)})
    small = _workspace(tmp_path, "small", {"x.md": "# Nota\nabc abc abc"})
    _indexer(store, emb)._index_sync(big, force=False, prune=True)
    _indexer(store, emb)._index_sync(small, force=False, prune=True)

    # The big workspace's chunks are nearer to this query, so a plain k=3 KNN
    # would only see them; the workspace filter must still return the note.
    query = emb.embed_query("zzzz qqqq abc")
    hits = store.vector_search(query, "small", 3)
    assert len(hits) == 1
    assert len(store.vector_search(query, None, 3)) == 3


def test_store_methods_serialize_on_one_connection(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db", vector_dim=DIM)
    emb = FakeEmbedder()
    spec = _workspace(tmp_path, "notes", {f"n{i}.md": f"# N{i}\ngatos {i}" for i in range(20)})
    _indexer(store, emb)._index_sync(spec, force=False, prune=True)
    query = emb.embed_query("gatos")
    errors: list[BaseException] = []

    def reader():
        try:
            for _ in range(200):
                store.bm25_search("gatos", None, 5)
                store.vector_search(query, None, 5)
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    def writer():
        try:
            for _ in range(10):
                _indexer(store, emb)._index_sync(spec, force=True, prune=True)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(3)] + [threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_changing_embedding_model_rebuilds_vectors(tmp_path):
    db = tmp_path / "kb.db"
    spec = _workspace(tmp_path, "notes", {"a.md": "# A\ngatos", "b.md": "# B\nperros"})
    store = KnowledgeStore(db, vector_dim=DIM, vector_model="model-a")
    _indexer(store, FakeEmbedder())._index_sync(spec, force=False, prune=True)
    n = _vec_count(store)
    assert n > 0
    store.close()

    # Same model and dim: vectors are kept, nothing to backfill.
    store = KnowledgeStore(db, vector_dim=DIM, vector_model="model-a")
    assert _vec_count(store) == n
    store.close()

    # No embedder this run: vectors are left alone.
    store = KnowledgeStore(db, vector_dim=384, vector_model=None)
    assert _vec_count(store) == n
    store.close()

    # Another model (even with the same dim): old vectors are dropped and
    # the next index run re-embeds every chunk.
    store = KnowledgeStore(db, vector_dim=DIM, vector_model="model-b")
    assert _vec_count(store) == 0
    report = _indexer(store, FakeEmbedder())._index_sync(spec, force=False, prune=True)
    assert report.vectors_backfilled == n == _vec_count(store)


def test_prefixes_follow_the_model_family():
    from eyetor.knowledge.embedder import _prefixes_for

    assert _prefixes_for("intfloat/multilingual-e5-large") == ("query: ", "passage: ")
    assert _prefixes_for("google/embeddinggemma-300m")[0].startswith("task: search result")
    assert _prefixes_for("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") == ("", "")
