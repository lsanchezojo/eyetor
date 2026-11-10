"""Cross-encoder reranking over an OpenAI/Jina-style ``/rerank`` endpoint.

BM25 and vector search each score a chunk on its own; a cross-encoder reads
the question and the chunk together, which is much better at telling which
passage actually answers. It is too slow to run over the whole KB, so the
retriever only sends it the top fused candidates.

Meant for ``llama-server --rerank`` (bge-reranker-v2-m3, Qwen3-Reranker…).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        base_url: str,
        *,
        model: str = "",
        api_key: str = "",
        timeout: float = 30.0,
        max_chars: int = 2000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_chars = max_chars

    @classmethod
    def from_config(cls, cfg) -> "Reranker | None":
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        if not getattr(cfg, "base_url", ""):
            logger.warning("knowledge.rerank.enabled needs base_url; reranking disabled")
            return None
        return cls(
            cfg.base_url,
            model=getattr(cfg, "model", ""),
            api_key=getattr(cfg, "api_key", ""),
            timeout=getattr(cfg, "timeout", 30.0),
            max_chars=getattr(cfg, "max_chars", 2000),
        )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score of each document, in the order given (sync)."""
        if not documents:
            return []
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "query": query,
            "documents": [d[: self.max_chars] for d in documents],
            "top_n": len(documents),
        }
        if self.model:
            payload["model"] = self.model
        resp = httpx.post(
            f"{self.base_url}/rerank", json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        scores = [float("-inf")] * len(documents)
        for item in resp.json()["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores
