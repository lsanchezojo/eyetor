"""Knowledge base module: hybrid BM25 + vector retrieval over workspace docs."""

from __future__ import annotations

from eyetor.knowledge.chunker import Chunker, chunk_document
from eyetor.knowledge.embedder import Embedder
from eyetor.knowledge.extractors import (
    ExtractedDoc,
    ExtractedSection,
    get_extractor,
    register_extractor,
    supported_extensions,
)
from eyetor.knowledge.indexer import IndexReport, Indexer, WorkspaceSpec
from eyetor.knowledge.manager import KnowledgeManager
from eyetor.knowledge.retriever import ReadResult, Retriever, SearchBackend, SearchHit, rrf_fuse
from eyetor.knowledge.store import Chunk, ChunkRow, DocRow, KnowledgeStore

__all__ = [
    "Chunk",
    "ChunkRow",
    "Chunker",
    "DocRow",
    "Embedder",
    "ExtractedDoc",
    "ExtractedSection",
    "IndexReport",
    "Indexer",
    "KnowledgeManager",
    "KnowledgeStore",
    "ReadResult",
    "Retriever",
    "SearchBackend",
    "SearchHit",
    "WorkspaceSpec",
    "chunk_document",
    "get_extractor",
    "register_extractor",
    "rrf_fuse",
    "supported_extensions",
]
