"""Embedder wrapper around fastembed for local, multilingual embeddings.

Uses ``paraphrase-multilingual-MiniLM-L12-v2`` (384 dims) by default. Models
trained with query/passage prefixes (E5, EmbeddingGemma) get them applied
automatically, see ``_PREFIXES``.

The wrapper is purposely tolerant: if ``fastembed`` is not installed,
``Embedder.from_config`` returns ``None`` and logs a single warning. The
rest of the knowledge stack detects the absence and falls back to BM25.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eyetor.config import KnowledgeEmbeddingConfig  # noqa: F401

logger = logging.getLogger(__name__)

_FASTEMBED_WARNING_LOGGED = False

# (query prefix, passage prefix) by model family. E5 and EmbeddingGemma are
# trained with them and lose quality without; other models (MiniLM, potion…)
# expect raw text, where a prefix is just noise.
_PREFIXES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("e5", ("query: ", "passage: ")),
    ("embeddinggemma", ("task: search result | query: ", "title: none | text: ")),
    (
        "qwen3-embedding",
        (
            "Instruct: Given a question, retrieve the passages of a document "
            "that answer it\nQuery: ",
            "",
        ),
    ),
)


def _prefixes_for(model_name: str) -> tuple[str, str]:
    name = model_name.lower()
    for family, prefixes in _PREFIXES:
        if family in name:
            return prefixes
    return "", ""


class Embedder:
    """Lazy wrapper over fastembed's TextEmbedding."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_dir: str | Path = "~/.eyetor/models/fastembed",
        dim: int = 384,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir).expanduser()
        self.dim = dim
        self.batch_size = batch_size
        self._model = None  # lazy
        self.query_prefix, self.passage_prefix = _prefixes_for(model_name)

    def set_prefixes(self, query: str | None, passage: str | None) -> None:
        """Override the per-family prefixes (None keeps the default)."""
        if query is not None:
            self.query_prefix = query
        if passage is not None:
            self.passage_prefix = passage

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "Embedder | None":
        """Create an Embedder from a ``KnowledgeEmbeddingConfig`` or ``None``.

        Returns ``None`` if disabled in config or if fastembed is not installed.
        """
        global _FASTEMBED_WARNING_LOGGED
        if cfg is None or not getattr(cfg, "enabled", True):
            return None
        if getattr(cfg, "backend", "fastembed") == "http":
            return HttpEmbedder.from_config(cfg)
        try:
            import fastembed  # type: ignore  # noqa: F401
        except ImportError:
            if not _FASTEMBED_WARNING_LOGGED:
                logger.warning(
                    "fastembed not installed, semantic retrieval disabled "
                    "(install eyetor[knowledge-vector])"
                )
                _FASTEMBED_WARNING_LOGGED = True
            return None
        emb = cls(
            model_name=getattr(cfg, "model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            model_dir=getattr(cfg, "model_dir", "~/.eyetor/models/fastembed"),
            dim=getattr(cfg, "dim", 384),
            batch_size=getattr(cfg, "batch_size", 64),
        )
        emb.set_prefixes(getattr(cfg, "query_prefix", None), getattr(cfg, "passage_prefix", None))
        return emb

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from fastembed import TextEmbedding  # type: ignore

        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loading fastembed model %s (cache=%s)", self.model_name, self.model_dir
        )
        self._model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(self.model_dir),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode a list of passages (with the model's passage prefix)."""
        if not texts:
            return []
        self._ensure_loaded()
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        vectors = list(self._model.embed(prefixed, batch_size=self.batch_size))  # type: ignore[union-attr]
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query (with the model's query prefix)."""
        if not text:
            return []
        self._ensure_loaded()
        vectors = list(self._model.embed([f"{self.query_prefix}{text}"], batch_size=1))  # type: ignore[union-attr]
        if not vectors:
            return []
        return [float(x) for x in vectors[0]]


class HttpEmbedder(Embedder):
    """Embeddings from an OpenAI-compatible ``/embeddings`` endpoint.

    Meant for ``llama-server --embedding`` (the model runs on the GPU in its
    own process, eyetor stays light). Calls are synchronous: callers already
    run them in a worker thread. Errors raise; the indexer logs and stores
    the doc without vectors (backfilled on the next run), and the retriever
    falls back to BM25 for that query.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        dim: int,
        *,
        batch_size: int = 32,
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model_name=model_name, dim=dim, batch_size=batch_size)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_config(cls, cfg) -> "HttpEmbedder | None":
        base_url = getattr(cfg, "base_url", "")
        if not base_url:
            logger.warning("knowledge.embedding.backend=http needs base_url; vectors disabled")
            return None
        emb = cls(
            base_url,
            model_name=getattr(cfg, "model", ""),
            dim=getattr(cfg, "dim", 384),
            batch_size=getattr(cfg, "batch_size", 32),
            api_key=getattr(cfg, "api_key", ""),
            timeout=getattr(cfg, "timeout", 120.0),
        )
        emb.set_prefixes(getattr(cfg, "query_prefix", None), getattr(cfg, "passage_prefix", None))
        return emb

    def _post(self, texts: list[str]) -> list[list[float]]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"input": texts}
        if self.model_name:
            payload["model"] = self.model_name
        resp = httpx.post(
            f"{self.base_url}/embeddings", json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        return [[float(x) for x in d["embedding"]] for d in data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        size = max(1, self.batch_size)
        for i in range(0, len(texts), size):
            out.extend(self._post([f"{self.passage_prefix}{t}" for t in texts[i : i + size]]))
        return out

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        vectors = self._post([f"{self.query_prefix}{text}"])
        return vectors[0] if vectors else []
