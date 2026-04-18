"""Chunker: ExtractedDoc → list of Chunks ready for indexing.

Two strategies:
- If ``ExtractedDoc.sections`` is populated (markdown, docx headings, pptx
  slides, xlsx sheets), respect section boundaries and propagate
  ``heading_path``. Oversized sections are subdivided by paragraphs.
- Otherwise fall back to paragraph splitting with ``max_chars`` target and
  ``overlap_chars`` overlap to preserve context between chunks.

For code files (``.py`` / ``.js`` / ``.ts`` / ``.go`` / ``.rs``), a soft
boundary regex prefers cuts at top-level declarations.
"""

from __future__ import annotations

import re
from pathlib import Path

from eyetor.knowledge.extractors import ExtractedDoc
from eyetor.knowledge.store import Chunk


_CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"}
_CODE_BOUNDARY_RE = re.compile(
    r"^(\s*)(def |class |function |export |func |fn |impl |public |private |protected )",
    re.MULTILINE,
)

# Matches the page markers that ``extract_pdf`` inserts, e.g. ``[Page 12]``.
_PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]")


class Chunker:
    def __init__(self, max_chars: int = 1500, overlap_chars: int = 150) -> None:
        self.max_chars = max_chars
        self.overlap_chars = max(0, overlap_chars)

    def chunk(self, doc: ExtractedDoc, suffix: str = "") -> list[Chunk]:
        """Main entry point."""
        if doc.sections:
            return self._chunk_by_sections(doc)
        if suffix.lower() in _CODE_EXTS:
            return self._chunk_code(doc)
        return self._chunk_paragraphs(doc)

    # ------------------------------------------------------------------
    # Structural chunking (markdown / docx / pptx / xlsx)
    # ------------------------------------------------------------------

    def _chunk_by_sections(self, doc: ExtractedDoc) -> list[Chunk]:
        lines = doc.text.splitlines()
        chunks: list[Chunk] = []
        ordinal = 0
        for section in doc.sections:
            start = max(0, section.start_line)
            end = min(len(lines) - 1, section.end_line)
            if end < start:
                continue
            section_text = "\n".join(lines[start : end + 1]).strip()
            if not section_text:
                continue
            for piece in self._split_oversized(section_text):
                chunks.append(
                    Chunk(
                        ordinal=ordinal,
                        heading_path=section.heading_path,
                        start_line=start + 1,
                        end_line=end + 1,
                        content=piece,
                    )
                )
                ordinal += 1
        if not chunks:
            # Fallback: section metadata was empty
            return self._chunk_paragraphs(doc)
        return chunks

    def _split_oversized(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        parts: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            para_len = len(para) + 2
            if buf and buf_len + para_len > self.max_chars:
                parts.append("\n\n".join(buf))
                if self.overlap_chars and parts[-1]:
                    tail = parts[-1][-self.overlap_chars :]
                    buf = [tail, para]
                    buf_len = len(tail) + para_len
                else:
                    buf = [para]
                    buf_len = para_len
            else:
                buf.append(para)
                buf_len += para_len
            while buf_len > self.max_chars and len(buf) == 1:
                # Single paragraph exceeds cap: hard split
                chunk = buf[0][: self.max_chars]
                parts.append(chunk)
                rest = buf[0][self.max_chars - self.overlap_chars :]
                buf = [rest]
                buf_len = len(rest)
        if buf:
            parts.append("\n\n".join(buf))
        return [p for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # Paragraph chunking (generic text)
    # ------------------------------------------------------------------

    def _chunk_paragraphs(self, doc: ExtractedDoc) -> list[Chunk]:
        chunks: list[Chunk] = []
        pieces = self._split_oversized(doc.text)
        title = doc.title or ""
        # Track the running page across pieces: if a piece has no marker of its
        # own it inherits the page from the previous piece (marker sits at the
        # top of the page, the rest of the page has no marker).
        current_page: int | None = None
        for i, piece in enumerate(pieces):
            match = _PAGE_MARKER_RE.search(piece)
            if match:
                current_page = int(match.group(1))
            heading: str | None
            if current_page is not None:
                heading = f"Page {current_page}"
            else:
                heading = title if title else None
            chunks.append(
                Chunk(
                    ordinal=i,
                    heading_path=heading,
                    start_line=None,
                    end_line=None,
                    content=piece,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Code chunking
    # ------------------------------------------------------------------

    def _chunk_code(self, doc: ExtractedDoc) -> list[Chunk]:
        text = doc.text
        if len(text) <= self.max_chars:
            return [
                Chunk(
                    ordinal=0,
                    heading_path=doc.title,
                    start_line=1,
                    end_line=text.count("\n") + 1,
                    content=text,
                )
            ]
        # Find preferred boundary offsets
        boundaries = [m.start() for m in _CODE_BOUNDARY_RE.finditer(text)]
        boundaries.append(len(text))
        chunks: list[Chunk] = []
        cursor = 0
        ordinal = 0
        while cursor < len(text):
            target = cursor + self.max_chars
            # pick the last boundary before `target`, or just cut at target
            cut = target
            for b in boundaries:
                if cursor < b <= target:
                    cut = b
                elif b > target:
                    break
            piece = text[cursor:cut].strip()
            if piece:
                start_line = text.count("\n", 0, cursor) + 1
                end_line = text.count("\n", 0, cut) + 1
                chunks.append(
                    Chunk(
                        ordinal=ordinal,
                        heading_path=doc.title,
                        start_line=start_line,
                        end_line=end_line,
                        content=piece,
                    )
                )
                ordinal += 1
            if cut <= cursor:
                cut = cursor + self.max_chars
            cursor = max(cut - self.overlap_chars, cut) if cut >= len(text) else cut
        return chunks


def chunk_document(
    doc: ExtractedDoc,
    *,
    path: str | Path = "",
    max_chars: int = 1500,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Convenience wrapper: build a Chunker and chunk a single doc."""
    suffix = Path(str(path)).suffix if path else ""
    chunker = Chunker(max_chars=max_chars, overlap_chars=overlap_chars)
    return chunker.chunk(doc, suffix=suffix)
