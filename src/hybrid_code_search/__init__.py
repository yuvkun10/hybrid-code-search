"""Public API for hybrid lexical and vector code search."""

from __future__ import annotations

from .chunker import chunk_file, chunk_paths, chunk_source, detect_language
from .embedder import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    resolve_embedder,
)
from .index import CodeIndex
from .ranking import LexicalIndex, fuse
from .store import load_index, save_index
from .tokenize import tokenize
from .types import Chunk, Embedder, SearchResult

__version__ = "0.1.0"


def build_index(paths: list[str], embedder: Embedder | None = None) -> CodeIndex:
    """Chunk the given paths and build a queryable index over them."""
    chunks = chunk_paths(paths)
    resolved = embedder if embedder is not None else resolve_embedder()
    return CodeIndex.build(chunks, resolved)


__all__ = [
    "Chunk",
    "CodeIndex",
    "Embedder",
    "HashingEmbedder",
    "LexicalIndex",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "__version__",
    "build_index",
    "chunk_file",
    "chunk_paths",
    "chunk_source",
    "detect_language",
    "fuse",
    "load_index",
    "resolve_embedder",
    "save_index",
    "tokenize",
]
