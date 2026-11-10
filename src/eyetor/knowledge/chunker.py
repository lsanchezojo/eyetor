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

# Bump when chunk boundaries change: KnowledgeManager.index_all re-chunks every
# document once when the stored version differs.
# 2: long PDF paragraphs were never split (chunks up to ~6k chars); page
#    headings took the first marker even when the chunk started on the
#    previous page.
CHUNKER_VERSION = 2

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
        """Pack ``text`` into pieces of at most ``max_chars`` (+ overlap).

        Units are paragraphs; a paragraph longer than ``max_chars`` (common in
        PDFs, where a whole page may have no blank line) is split by lines,
        then sentences, then at a space. Each new piece starts with the tail
        of the previous one (``overlap_chars``, cut at a word boundary).
        """
        if len(text) <= self.max_chars:
            return [text]
        units: list[str] = []
        for para in re.split(r"\n\s*\n", text):
            units.extend(self._fit_units(para.strip()))
        parts: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for unit in units:
            if not unit:
                continue
            sep = 2 if buf else 0
            if buf and buf_len + sep + len(unit) > self.max_chars:
                parts.append("\n\n".join(buf))
                tail = self._overlap_tail(parts[-1])
                buf = [tail] if tail else []
                buf_len = len(tail)
                sep = 2 if buf else 0
            buf.append(unit)
            buf_len += sep + len(unit)
        if buf:
            parts.append("\n\n".join(buf))
        return [p for p in parts if p.strip()]

    def _fit_units(self, text: str) -> list[str]:
        """Split ``text`` into units no longer than ``max_chars``."""
        if len(text) <= self.max_chars:
            return [text]
        for pattern in (r"\n", r"(?<=[.!?:;])\s+"):
            pieces = [p.strip() for p in re.split(pattern, text) if p.strip()]
            if len(pieces) > 1:
                return self._pack(pieces, " " if pattern != r"\n" else "\n")
        # One huge run without breaks: cut at the last space before the cap.
        out: list[str] = []
        rest = text
        while len(rest) > self.max_chars:
            cut = rest.rfind(" ", 0, self.max_chars)
            if cut <= 0:
                cut = self.max_chars
            out.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest:
            out.append(rest)
        return out

    def _pack(self, pieces: list[str], joiner: str) -> list[str]:
        """Greedily join ``pieces`` into units of at most ``max_chars``."""
        out: list[str] = []
        cur = ""
        for piece in pieces:
            for sub in self._fit_units(piece) if len(piece) > self.max_chars else [piece]:
                if cur and len(cur) + len(joiner) + len(sub) > self.max_chars:
                    out.append(cur)
                    cur = sub
                else:
                    cur = f"{cur}{joiner}{sub}" if cur else sub
        if cur:
            out.append(cur)
        return out

    def _overlap_tail(self, text: str) -> str:
        if not self.overlap_chars or len(text) <= self.overlap_chars:
            return ""
        tail = text[-self.overlap_chars :]
        space = tail.find(" ")
        return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail

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
            # The heading is the page the piece starts on: text before the
            # first marker still belongs to the previous page.
            markers = list(_PAGE_MARKER_RE.finditer(piece))
            start_page = current_page
            if markers and (start_page is None or not piece[: markers[0].start()].strip()):
                start_page = int(markers[0].group(1))
            if markers:
                current_page = int(markers[-1].group(1))
            heading: str | None
            if start_page is not None:
                heading = f"Page {start_page}"
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
