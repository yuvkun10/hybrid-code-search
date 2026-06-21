"""Core data types shared across the package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Chunk:
    """A semantically meaningful span of source code."""

    id: str
    path: str
    language: str
    kind: str  # one of: function, class, method, module, block
    symbol: str  # qualified name, or "" for module-level/anonymous blocks
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A chunk ranked against a query."""

    chunk: Chunk
    score: float  # fused score in [0, 1]
    vector_score: float  # cosine similarity in [-1, 1]
    lexical_score: float  # normalized lexical score in [0, 1]


@runtime_checkable
class Embedder(Protocol):
    """Maps text to fixed-dimension, L2-normalized vectors.

    Implementations must be deterministic: the same text always yields the
    same vector, so an index can be persisted and reloaded without drift.
    """

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalized vector per input text."""
        ...
