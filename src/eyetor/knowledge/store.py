"""SQLite-backed store for the Knowledge Base.

Hybrid retrieval backend:
- FTS5 (stdlib) for BM25 lexical search.
- sqlite-vec (optional extension) for semantic vector search.

If sqlite-vec is unavailable, the store falls back transparently to BM25-only:
``vector_search`` returns an empty list and ``replace_chunks`` ignores embeddings.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


_FTS5_EXPLICIT_OPS = {"AND", "OR", "NOT", "NEAR"}
_WORDLIKE_RE = re.compile(r"^[\w]+$", re.UNICODE)


def sanitize_fts5_query(query: str) -> str:
    """Make a natural-language query safe for FTS5.

    FTS5 treats ``-`` as NOT, ``.``/``/`` as separators, and a bare ``foo-bar``
    query parses as ``foo NOT bar``. If the user passes an explicit FTS5
    expression (uppercase AND/OR/NOT/NEAR, quotes, parens, ``*``, column
    filter ``col:``), we respect it verbatim. Otherwise every token that isn't
    purely alphanumeric gets quoted as a literal phrase, so
    ``two-phase compaction`` becomes ``"two-phase" compaction``.

    Tokens are joined with ``OR`` rather than the FTS5 implicit-AND so a
    natural-language query like ``RE acronym second era`` can still surface
    chunks that only contain a subset of terms (common when the document
    language differs from the query language). BM25 handles ranking — chunks
    matching more terms score higher.
    """
    q = (query or "").strip()
    if not q:
        return q

    tokens = q.split()
    has_explicit_op = any(t in _FTS5_EXPLICIT_OPS for t in tokens)
    if (
        has_explicit_op
        or '"' in q
        or "(" in q
        or ")" in q
        or "*" in q
        or ":" in q
    ):
        return q

    out: list[str] = []
    for tok in tokens:
        stripped = tok.strip(".,;:!?")
        if not stripped:
            continue
        if _WORDLIKE_RE.match(stripped):
            out.append(stripped)
            continue
        escaped = stripped.replace('"', '""')
        out.append(f'"{escaped}"')
    return " OR ".join(out)

_SQLITE_VEC_WARNING_LOGGED = False


@dataclass
class Chunk:
    """A single chunk of text extracted from a document."""

    ordinal: int
    heading_path: str | None
    start_line: int | None
    end_line: int | None
    content: str


@dataclass
class ChunkRow:
    """A chunk row hydrated from the store with doc metadata."""

    chunk_id: int
    doc_id: int
    workspace: str
    rel_path: str
    title: str | None
    heading_path: str | None
    content: str


@dataclass
class DocRow:
    """A document row from the store."""

    id: int
    workspace: str
    rel_path: str
    abs_path: str
    title: str | None
    mtime: float
    size_bytes: int
    sha1: str
    ext: str
    last_indexed: str


_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS workspaces (
    name        TEXT PRIMARY KEY,
    root_path   TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS docs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace    TEXT NOT NULL REFERENCES workspaces(name) ON DELETE CASCADE,
    rel_path     TEXT NOT NULL,
    abs_path     TEXT NOT NULL,
    title        TEXT,
    mtime        REAL NOT NULL,
    size_bytes   INTEGER NOT NULL,
    sha1         TEXT NOT NULL,
    ext          TEXT NOT NULL,
    last_indexed TEXT NOT NULL,
    UNIQUE (workspace, rel_path)
);
CREATE INDEX IF NOT EXISTS idx_docs_workspace ON docs(workspace);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id        INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    heading_path  TEXT,
    start_line    INTEGER,
    end_line      INTEGER,
    content       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading_path,
    content,
    content='chunks',
    content_rowid='id',
    tokenize = "unicode61 remove_diacritics 2 tokenchars '_-.'"
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, heading_path, content)
    VALUES (new.id, new.heading_path, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading_path, content)
    VALUES ('delete', old.id, old.heading_path, old.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading_path, content)
    VALUES ('delete', old.id, old.heading_path, old.content);
    INSERT INTO chunks_fts(rowid, heading_path, content)
    VALUES (new.id, new.heading_path, new.content);
END;
"""


def _encode_vector(vec: list[float]) -> bytes:
    """Pack a float vector as little-endian float32 bytes (sqlite-vec format)."""
    return struct.pack(f"{len(vec)}f", *vec)


class KnowledgeStore:
    """SQLite store for knowledge-base workspaces/docs/chunks with FTS5 + vec."""

    def __init__(self, db_path: str | Path, *, vector_dim: int = 384) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._vector_dim = vector_dim
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.vector_enabled = self._try_load_sqlite_vec()
        self._conn.executescript(_DDL)
        if self.vector_enabled:
            self._create_vec_table()
        self._conn.commit()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, db_path: str | Path, *, vector_dim: int = 384) -> "KnowledgeStore":
        return cls(db_path, vector_dim=vector_dim)

    def _try_load_sqlite_vec(self) -> bool:
        global _SQLITE_VEC_WARNING_LOGGED
        try:
            import sqlite_vec  # type: ignore
        except ImportError:
            if not _SQLITE_VEC_WARNING_LOGGED:
                logger.warning(
                    "sqlite-vec unavailable, falling back to BM25-only retrieval "
                    "(install eyetor[knowledge-vector])"
                )
                _SQLITE_VEC_WARNING_LOGGED = True
            return False
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except (sqlite3.OperationalError, AttributeError) as exc:
            if not _SQLITE_VEC_WARNING_LOGGED:
                logger.warning(
                    "sqlite-vec failed to load (%s); using BM25-only retrieval", exc
                )
                _SQLITE_VEC_WARNING_LOGGED = True
            return False

    def _create_vec_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self._vector_dim}]
            )
            """
        )

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    def upsert_workspace(self, name: str, root_path: str) -> None:
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            INSERT INTO workspaces (name, root_path, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET root_path = excluded.root_path
            """,
            (name, root_path, now),
        )
        self._conn.commit()

    def list_workspaces(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT name, root_path, created_at FROM workspaces ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_workspace(self, name: str) -> None:
        # Collect chunk ids before cascade so we can clean the vec table.
        chunk_ids = [
            r["id"]
            for r in self._conn.execute(
                """
                SELECT c.id FROM chunks c
                JOIN docs d ON d.id = c.doc_id
                WHERE d.workspace = ?
                """,
                (name,),
            ).fetchall()
        ]
        self._delete_vec_rows(chunk_ids)
        self._conn.execute("DELETE FROM workspaces WHERE name = ?", (name,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Docs
    # ------------------------------------------------------------------

    def get_doc(self, workspace: str, rel_path: str) -> DocRow | None:
        row = self._conn.execute(
            "SELECT * FROM docs WHERE workspace = ? AND rel_path = ?",
            (workspace, rel_path),
        ).fetchone()
        return DocRow(**dict(row)) if row else None

    def get_doc_by_id(self, doc_id: int) -> DocRow | None:
        row = self._conn.execute(
            "SELECT * FROM docs WHERE id = ?", (doc_id,)
        ).fetchone()
        return DocRow(**dict(row)) if row else None

    def doc_exists(self, doc_id: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM docs WHERE id = ? LIMIT 1", (doc_id,)
        ).fetchone()
        return row is not None

    def list_docs(
        self, workspace: str | None = None, limit: int = 100
    ) -> list[DocRow]:
        if workspace:
            rows = self._conn.execute(
                "SELECT * FROM docs WHERE workspace = ? ORDER BY rel_path LIMIT ?",
                (workspace, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM docs ORDER BY workspace, rel_path LIMIT ?", (limit,)
            ).fetchall()
        return [DocRow(**dict(r)) for r in rows]

    def upsert_doc(
        self,
        workspace: str,
        rel_path: str,
        abs_path: str,
        title: str | None,
        mtime: float,
        size_bytes: int,
        sha1: str,
        ext: str,
    ) -> int:
        now = datetime.utcnow().isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO docs
              (workspace, rel_path, abs_path, title, mtime, size_bytes, sha1, ext, last_indexed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace, rel_path) DO UPDATE SET
                abs_path     = excluded.abs_path,
                title        = excluded.title,
                mtime        = excluded.mtime,
                size_bytes   = excluded.size_bytes,
                sha1         = excluded.sha1,
                ext          = excluded.ext,
                last_indexed = excluded.last_indexed
            RETURNING id
            """,
            (workspace, rel_path, abs_path, title, mtime, size_bytes, sha1, ext, now),
        )
        doc_id = cur.fetchone()[0]
        self._conn.commit()
        return doc_id

    def delete_doc(self, doc_id: int) -> None:
        chunk_ids = [
            r["id"]
            for r in self._conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchall()
        ]
        self._delete_vec_rows(chunk_ids)
        self._conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
        self._conn.commit()

    def all_doc_rel_paths(self, workspace: str) -> list[tuple[int, str]]:
        rows = self._conn.execute(
            "SELECT id, rel_path FROM docs WHERE workspace = ?", (workspace,)
        ).fetchall()
        return [(r["id"], r["rel_path"]) for r in rows]

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def replace_chunks(
        self,
        doc_id: int,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> list[int]:
        """Replace all chunks of a doc. Returns new chunk ids in order.

        If ``embeddings`` is provided and vector search is enabled, the chunks
        are inserted into the vec table in the same order.
        """
        old_ids = [
            r["id"]
            for r in self._conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchall()
        ]
        self._delete_vec_rows(old_ids)
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

        new_ids: list[int] = []
        for ch in chunks:
            cur = self._conn.execute(
                """
                INSERT INTO chunks
                  (doc_id, ordinal, heading_path, start_line, end_line, content)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, ch.ordinal, ch.heading_path, ch.start_line, ch.end_line, ch.content),
            )
            new_ids.append(cur.lastrowid)

        if (
            embeddings
            and self.vector_enabled
            and len(embeddings) == len(new_ids)
        ):
            for chunk_id, emb in zip(new_ids, embeddings):
                if len(emb) != self._vector_dim:
                    logger.warning(
                        "Skipping embedding for chunk %d: expected dim %d, got %d",
                        chunk_id,
                        self._vector_dim,
                        len(emb),
                    )
                    continue
                self._conn.execute(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, _encode_vector(emb)),
                )

        self._conn.commit()
        return new_ids

    def _delete_vec_rows(self, chunk_ids: list[int]) -> None:
        if not chunk_ids or not self.vector_enabled:
            return
        placeholders = ",".join("?" * len(chunk_ids))
        try:
            self._conn.execute(
                f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
        except sqlite3.OperationalError as exc:
            logger.debug("vec delete ignored: %s", exc)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def bm25_search(
        self, query: str, workspace: str | None, k: int
    ) -> list[int]:
        """Return chunk ids ranked by BM25 relevance."""
        fts_query = sanitize_fts5_query(query)
        if not fts_query:
            return []
        sql = """
            SELECT c.id AS id
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN docs   d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ?
              AND (? IS NULL OR d.workspace = ?)
            ORDER BY bm25(chunks_fts, 2.0, 1.0)
            LIMIT ?
        """
        try:
            rows = self._conn.execute(
                sql, (fts_query, workspace, workspace, k)
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("bm25_search failed for query %r: %s", fts_query, exc)
            return []
        return [r["id"] for r in rows]

    def vector_search(
        self, query_embedding: list[float], workspace: str | None, k: int
    ) -> list[int]:
        """Return chunk ids ranked by cosine distance via sqlite-vec KNN."""
        if not self.vector_enabled or not query_embedding:
            return []
        if len(query_embedding) != self._vector_dim:
            logger.warning(
                "vector_search: expected dim %d, got %d",
                self._vector_dim,
                len(query_embedding),
            )
            return []
        vec_blob = _encode_vector(query_embedding)
        # sqlite-vec's KNN uses `embedding MATCH ? AND k = ?`.
        sql = """
            SELECT c.id AS id
            FROM chunks_vec
            JOIN chunks c ON c.id = chunks_vec.chunk_id
            JOIN docs   d ON d.id = c.doc_id
            WHERE chunks_vec.embedding MATCH ?
              AND k = ?
              AND (? IS NULL OR d.workspace = ?)
            ORDER BY distance
        """
        try:
            rows = self._conn.execute(
                sql, (vec_blob, k, workspace, workspace)
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("vector_search failed: %s", exc)
            return []
        return [r["id"] for r in rows]

    def fetch_chunks(self, chunk_ids: list[int]) -> list[ChunkRow]:
        """Hydrate a list of chunk ids with their content and doc metadata."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"""
            SELECT c.id AS chunk_id, c.doc_id, c.heading_path, c.content,
                   d.workspace, d.rel_path, d.title
            FROM chunks c
            JOIN docs d ON d.id = c.doc_id
            WHERE c.id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        ordered: list[ChunkRow] = []
        for cid in chunk_ids:
            r = by_id.get(cid)
            if not r:
                continue
            ordered.append(
                ChunkRow(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    workspace=r["workspace"],
                    rel_path=r["rel_path"],
                    title=r["title"],
                    heading_path=r["heading_path"],
                    content=r["content"],
                )
            )
        return ordered

    def read_chunks(
        self, doc_id: int, heading_prefix: str | None = None
    ) -> list[ChunkRow]:
        """Return chunks of a doc in order, optionally filtered by heading.

        The ``heading_prefix`` name is historical: matching is now a
        case-insensitive substring so the LLM can pass an approximate heading
        ("líneas de habilidad" matches "Líneas de habilidad", "3. Líneas de
        habilidad", etc.). This is important for non-structural extractors
        (PDFs) where the LLM rarely gets the heading verbatim.
        """
        if heading_prefix:
            rows = self._conn.execute(
                """
                SELECT c.id AS chunk_id, c.doc_id, c.heading_path, c.content,
                       d.workspace, d.rel_path, d.title
                FROM chunks c
                JOIN docs d ON d.id = c.doc_id
                WHERE c.doc_id = ?
                  AND c.heading_path LIKE ? COLLATE NOCASE
                ORDER BY c.ordinal
                """,
                (doc_id, f"%{heading_prefix}%"),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT c.id AS chunk_id, c.doc_id, c.heading_path, c.content,
                       d.workspace, d.rel_path, d.title
                FROM chunks c
                JOIN docs d ON d.id = c.doc_id
                WHERE c.doc_id = ?
                ORDER BY c.ordinal
                """,
                (doc_id,),
            ).fetchall()
        return [
            ChunkRow(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                workspace=r["workspace"],
                rel_path=r["rel_path"],
                title=r["title"],
                heading_path=r["heading_path"],
                content=r["content"],
            )
            for r in rows
        ]

    def list_sections(self, doc_id: int, limit: int = 20) -> list[str]:
        """Return distinct heading_path values for a doc, ordered by first ordinal."""
        rows = self._conn.execute(
            """
            SELECT heading_path, MIN(ordinal) AS first_ordinal
            FROM chunks
            WHERE doc_id = ? AND heading_path IS NOT NULL
            GROUP BY heading_path
            ORDER BY first_ordinal
            LIMIT ?
            """,
            (doc_id, limit),
        ).fetchall()
        return [r["heading_path"] for r in rows if r["heading_path"]]

    def snippet_for(self, chunk_id: int, query: str, max_tokens: int = 32) -> str:
        """Use FTS5 snippet() for a single chunk id; fall back to raw content."""
        fts_query = sanitize_fts5_query(query)
        try:
            row = self._conn.execute(
                f"""
                SELECT snippet(chunks_fts, 1, '<<', '>>', '…', {int(max_tokens)}) AS snip
                FROM chunks_fts
                WHERE chunks_fts.rowid = ? AND chunks_fts MATCH ?
                """,
                (chunk_id, fts_query),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        if row and row["snip"]:
            return row["snip"]
        # Fallback: first N chars of the content
        row = self._conn.execute(
            "SELECT content FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return ""
        text = row["content"]
        return text[:400] + ("…" if len(text) > 400 else "")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        ws = self._conn.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0]
        docs = self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        try:
            size = self._path.stat().st_size
        except OSError:
            size = 0
        return {
            "workspaces": ws,
            "docs": docs,
            "chunks": chunks,
            "vector_enabled": self.vector_enabled,
            "db_size_bytes": size,
            "db_path": str(self._path),
        }

    def close(self) -> None:
        self._conn.close()
