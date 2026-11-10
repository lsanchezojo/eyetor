"""Chunk sizes: long PDF-like paragraphs must be split, page headings exact."""

from __future__ import annotations

from eyetor.knowledge.chunker import Chunker
from eyetor.knowledge.extractors import ExtractedDoc


def _doc(text: str) -> ExtractedDoc:
    return ExtractedDoc(text=text, title="t")


def test_pages_without_blank_lines_are_split_to_max_chars():
    # Each "page" is one long paragraph of single-newline lines, as pypdf gives.
    page = "\n".join(f"Línea {i} con algo de texto de relleno para ocupar sitio." for i in range(80))
    text = "\n\n".join(f"[Page {n}]\n{page}" for n in range(1, 4))
    chunker = Chunker(max_chars=1000, overlap_chars=100)
    chunks = chunker.chunk(_doc(text))
    assert len(chunks) > 3
    assert max(len(c.content) for c in chunks) <= 1000 + 100 + 2


def test_sentence_and_space_fallbacks():
    chunker = Chunker(max_chars=200, overlap_chars=20)
    sentences = " ".join(f"Frase número {i} del texto." for i in range(60))
    pieces = chunker._split_oversized(sentences)
    assert all(len(p) <= 222 for p in pieces)
    no_breaks = "palabra " * 200
    assert all(len(p) <= 222 for p in chunker._split_oversized(no_breaks))
    assert all(not p.startswith("alabra") for p in chunker._split_oversized(no_breaks))


def test_heading_is_the_page_a_chunk_starts_on():
    body = "\n".join(f"texto de la página línea {i}" for i in range(30))
    text = f"[Page 1]\n{body}\n[Page 2]\n{body}"
    chunks = Chunker(max_chars=500, overlap_chars=0).chunk(_doc(text))
    for c in chunks:
        first = c.content.lstrip()
        if first.startswith("[Page "):
            assert c.heading_path == f"Page {first[6]}"
    # A chunk that starts with page-1 text and contains the page-2 marker is page 1.
    mixed = [c for c in chunks if "[Page 2]" in c.content and not c.content.lstrip().startswith("[Page 2]")]
    assert mixed and all(c.heading_path == "Page 1" for c in mixed)
