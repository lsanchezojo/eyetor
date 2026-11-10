"""KB retrieval eval: do the top-k chunks contain the passage that answers?

Runs each question through ``KnowledgeManager.search`` (the same path the
``kb_search`` tool uses) and checks whether any of the top-k chunks contains
one of the case's needle phrases (case-insensitive, whitespace collapsed).

Usage (from the repo root):

    .venv/bin/python evals/kb_retrieval.py [CASES.yaml] [-k 5] [--db PATH]
        [--set embedding.enabled=true --set rerank.enabled=false ...]
        [--reindex] [-v]

CASES defaults to ~/.eyetor/evals/kb.yaml (your own documents, kept out of
the repo), falling back to evals/kb.example.yaml. ``--set`` overrides keys of
the ``knowledge`` config section (dotted paths, YAML values), so variants can
be compared without editing config files. ``--db`` points at a copy of the
knowledge database (indexing a new embedding model writes vectors into it);
``--reindex`` re-chunks every document (needed after chunk.* changes).

Case format (YAML list):

    - q: "¿qué pasa si mi personaje cae a cero puntos de golpe?"
      needles: ["tres tiradas de muerte"]   # any of them, in one top-k chunk
      workspace: default                    # optional
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import statistics
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
LOCAL_CASES = Path.home() / ".eyetor/evals/kb.yaml"
EXAMPLE_CASES = REPO / "evals/kb.example.yaml"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def first_hit_rank(case: dict, contents: list[str]) -> int | None:
    """1-based rank of the first chunk containing a needle, or None."""
    needles = [_norm(n) for n in case["needles"]]
    for rank, content in enumerate(contents, 1):
        text = _norm(content)
        if any(n in text for n in needles):
            return rank
    return None


def apply_overrides(cfg, overrides: list[str]) -> None:
    """``a.b=value`` → setattr(cfg.a, "b", yaml(value))."""
    for item in overrides:
        key, _, raw = item.partition("=")
        *path, leaf = key.strip().split(".")
        target = cfg
        for part in path:
            target = getattr(target, part)
        if not hasattr(target, leaf):
            raise SystemExit(f"--set: unknown knowledge key {key!r}")
        setattr(target, leaf, yaml.safe_load(raw))


async def run(
    cases: list[dict], k: int, db: str | None, overrides: list[str], verbose: bool, reindex: bool
) -> int:
    from eyetor.config import load_config
    from eyetor.knowledge.manager import KnowledgeManager

    cfg = load_config().knowledge
    if db:
        cfg.db_path = db
    apply_overrides(cfg, overrides)
    km = KnowledgeManager.from_config(cfg)
    # New chunks/models get their vectors here (no-op when up to date).
    t0 = time.monotonic()
    reports = await km.index_all(force=reindex)
    backfilled = sum(r.vectors_backfilled for r in reports.values())
    written = sum(r.chunks_written for r in reports.values())
    if backfilled or written:
        print(f"(indexed {written} chunks, {backfilled} backfilled vectors, {time.monotonic() - t0:.0f}s)")

    ranks: list[int | None] = []
    latencies: list[float] = []
    for case in cases:
        t = time.monotonic()
        hits = await km.search(case["q"], workspace=case.get("workspace"), top_k=k)
        latencies.append(time.monotonic() - t)
        rows = {r.chunk_id: r.content for r in km.store.fetch_chunks([h.chunk_id for h in hits])}
        rank = first_hit_rank(case, [rows.get(h.chunk_id, "") for h in hits])
        ranks.append(rank)
        mark = "✅" if rank == 1 else ("🟡" if rank else "❌")
        print(f"{mark} {rank or '-':>2}  {case['q'][:70]}")
        if verbose and rank != 1:
            for h in hits[:3]:
                print(f"        {h.sources} {h.path[:30]} {h.heading or ''}: {_norm(h.snippet)[:90]}")

    n = len(cases)
    found = [r for r in ranks if r]
    mrr = sum(1 / r for r in found) / n
    print(
        f"\nhit@1 {sum(r == 1 for r in found)}/{n}  hit@3 {sum(r <= 3 for r in found)}/{n}  "
        f"hit@{k} {len(found)}/{n}  MRR {mrr:.3f}  "
        f"latencia mediana {statistics.median(latencies) * 1000:.0f} ms"
    )
    km.store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="?", type=Path)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--db", help="Knowledge DB to use (e.g. a copy)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="Override a knowledge config key, e.g. embedding.enabled=true")
    parser.add_argument("--reindex", action="store_true", help="Re-chunk and re-embed every document")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show top hits of misses")
    args = parser.parse_args()
    path = args.cases or (LOCAL_CASES if LOCAL_CASES.exists() else EXAMPLE_CASES)
    cases = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    print(f"cases: {path}")
    logging.basicConfig(level=logging.WARNING)
    return asyncio.run(run(cases, args.top_k, args.db, args.overrides, args.verbose, args.reindex))


if __name__ == "__main__":
    sys.exit(main())
