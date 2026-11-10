"""KB retrieval pipeline: FTS query building, reranking, HTTP embedder,
re-chunking on chunker version change. No model or server involved."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from eyetor.config import KnowledgeConfig, KnowledgeWorkspaceConfig
from eyetor.knowledge import store as store_mod
from eyetor.knowledge.chunker import CHUNKER_VERSION
from eyetor.knowledge.embedder import Embedder, HttpEmbedder
from eyetor.knowledge.manager import KnowledgeManager
from eyetor.knowledge.reranker import Reranker
from eyetor.knowledge.retriever import Retriever
from eyetor.knowledge.store import sanitize_fts5_query


# --- FTS query ---------------------------------------------------------------


def test_stopwords_dropped_and_words_stemmed_to_prefixes():
    q = sanitize_fts5_query("¿cómo recupero los puntos de voluntad?")
    terms = q.split(" OR ")
    assert "los" not in terms and "de" not in terms and "como" not in terms
    if store_mod._STEMMER is not None:
        assert "recuper*" in terms and "volunt*" in terms
    else:  # pragma: no cover
        assert "recupero" in terms


def test_explicit_fts_syntax_is_kept_verbatim():
    assert sanitize_fts5_query('"tirada de muerte" AND dragón') == '"tirada de muerte" AND dragón'
    assert sanitize_fts5_query("title:dragon") == "title:dragon"


def test_colon_in_natural_language_is_not_a_column_filter():
    q = sanitize_fts5_query("en Trey: ¿cómo funciona el oráculo?")
    assert " OR " in q and "trey" in q


def test_only_stopwords_still_searches_something():
    assert sanitize_fts5_query("de la") != ""


def test_hyphenated_tokens_stay_quoted():
    assert '"two-phase"' in sanitize_fts5_query("two-phase compaction")


# --- Reranking ---------------------------------------------------------------


class _Row:
    def __init__(self, cid: int, content: str) -> None:
        self.chunk_id = cid
        self.doc_id = 1
        self.workspace = "w"
        self.rel_path = "doc.md"
        self.title = None
        self.heading_path = None
        self.content = content


class _Backend:
    vector_enabled = False

    def __init__(self, contents: dict[int, str]) -> None:
        self.contents = contents

    def bm25_search(self, query, workspace, k):
        return list(self.contents)[:k]  # fixed order: 1, 2, 3

    def vector_search(self, *a):
        return []

    def fetch_chunks(self, ids):
        return [_Row(i, self.contents[i]) for i in ids if i in self.contents]

    def snippet_for(self, cid, query, n):
        return self.contents[cid]


class _FakeReranker:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.seen: list[str] = []

    def rerank(self, query, documents):
        if self.fail:
            raise httpx.ConnectError("down")
        self.seen = documents
        return [1.0 if "respuesta" in d else 0.1 for d in documents]


def test_reranker_reorders_candidates():
    backend = _Backend({1: "ruido", 2: "más ruido", 3: "aquí está la respuesta"})
    rr = _FakeReranker()
    hits = asyncio.run(Retriever(backend, reranker=rr, rerank_candidates=3).search("q", top_k=2))
    assert [h.chunk_id for h in hits] == [3, 1]
    assert "rerank" in hits[0].sources and hits[0].score == 1.0
    assert len(rr.seen) == 3


def test_reranker_failure_keeps_fused_order():
    backend = _Backend({1: "a", 2: "b", 3: "respuesta"})
    hits = asyncio.run(
        Retriever(backend, reranker=_FakeReranker(fail=True), rerank_candidates=3).search("q", top_k=2)
    )
    assert [h.chunk_id for h in hits] == [1, 2]


def test_reranker_client_maps_scores_back_to_document_order(monkeypatch):
    def fake_post(url, json, headers, timeout):
        assert url.endswith("/rerank") and json["documents"] == ["aaa", "bb"]
        return httpx.Response(
            200,
            json={"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    rr = Reranker("http://x/v1", max_chars=3)
    assert rr.rerank("q", ["aaaa", "bb"]) == [0.2, 0.9]


# --- HTTP embedder -----------------------------------------------------------


def test_http_embedder_batches_and_applies_qwen3_prefixes(monkeypatch):
    calls: list[list[str]] = []

    def fake_post(url, json, headers, timeout):
        calls.append(json["input"])
        data = [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(json["input"]))]
        return httpx.Response(200, json={"data": data}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    emb = HttpEmbedder("http://x/v1", model_name="Qwen3-Embedding-0.6B", dim=2, batch_size=2)
    assert len(emb.embed_documents(["a", "b", "c"])) == 3
    assert calls[0] == ["a", "b"] and calls[1] == ["c"]  # passages: no prefix
    emb.embed_query("¿hola?")
    assert calls[-1][0].startswith("Instruct:") and calls[-1][0].endswith("Query: ¿hola?")


def test_from_config_picks_backend_and_prefix_overrides():
    cfg = KnowledgeConfig().embedding
    cfg.backend, cfg.base_url, cfg.model, cfg.query_prefix = "http", "http://x/v1", "m", "Q: "
    emb = Embedder.from_config(cfg)
    assert isinstance(emb, HttpEmbedder) and emb.query_prefix == "Q: "
    cfg.base_url = ""
    assert Embedder.from_config(cfg) is None


# --- Re-chunk on chunker version change --------------------------------------


def test_chunker_version_change_rechunks_once(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# A\n\ntexto", encoding="utf-8")
    cfg = KnowledgeConfig(
        enabled=True,
        db_path=str(tmp_path / "kb.db"),
        workspaces=[KnowledgeWorkspaceConfig(name="w", path=str(root), include=["*.md"])],
    )
    cfg.embedding.enabled = False

    km = KnowledgeManager.from_config(cfg)
    assert asyncio.run(km.index_all())["w"].indexed == 1
    assert km.store.get_meta("chunker_version") == str(CHUNKER_VERSION)
    assert asyncio.run(km.index_all())["w"].skipped == 1  # unchanged, same version

    km.store.set_meta("chunker_version", "1")  # as left by older code
    assert asyncio.run(km.index_all())["w"].updated == 1  # re-chunked
    assert asyncio.run(km.index_all())["w"].skipped == 1
    km.store.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_rerank_disabled_or_without_url_builds_no_reranker(enabled):
    cfg = KnowledgeConfig().rerank
    cfg.enabled = enabled
    assert Reranker.from_config(cfg) is None
