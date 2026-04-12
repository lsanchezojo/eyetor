"""Embedder wrapper around fastembed for local, multilingual embeddings.

Uses ``intfloat/multilingual-e5-small`` (384 dims) by default. The E5 family
requires ``"query: "`` / ``"passage: "`` prefixes — omitting them measurably
degrades retrieval quality, so they are applied automatically here.

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


class Embedder:
    """Lazy wrapper over fastembed's TextEmbedding."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        model_dir: str | Path = "~/.eyetor/models/fastembed",
        dim: int = 384,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir).expanduser()
        self.dim = dim
        self.batch_size = batch_size
        self._model = None  # lazy

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
        return cls(
            model_name=getattr(cfg, "model", "intfloat/multilingual-e5-small"),
            model_dir=getattr(cfg, "model_dir", "~/.eyetor/models/fastembed"),
            dim=getattr(cfg, "dim", 384),
            batch_size=getattr(cfg, "batch_size", 64),
        )

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
        """Batch-encode a list of passages. Prepends the E5 ``passage:`` prefix."""
        if not texts:
            return []
        self._ensure_loaded()
        prefixed = [f"passage: {t}" for t in texts]
        vectors = list(self._model.embed(prefixed, batch_size=self.batch_size))  # type: ignore[union-attr]
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query with the E5 ``query:`` prefix."""
        if not text:
            return []
        self._ensure_loaded()
        vectors = list(self._model.embed([f"query: {text}"], batch_size=1))  # type: ignore[union-attr]
        if not vectors:
            return []
        return [float(x) for x in vectors[0]]
