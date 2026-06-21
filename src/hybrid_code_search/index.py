"""In-memory hybrid index combining dense vectors with lexical scoring."""

from __future__ import annotations

import numpy as np

from .embedder import HashingEmbedder, resolve_embedder
from .ranking import LexicalIndex, fuse, min_max_normalize
from .types import Chunk, Embedder, SearchResult


def _as_normalized_matrix(vectors: list[list[float]], dim: int) -> np.ndarray:
    """Stack embedding rows into a float32 matrix and L2-normalize each row.

    Re-normalizing here keeps cosine similarity correct even if an embedder
    returns rows with minor numerical drift, and gives a stable shape for an
    empty index so downstream code can rely on a 2-D array.
    """
    if not vectors:
        return np.zeros((0, dim), dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for all-zero rows; their cosine stays 0.
    norms[norms == 0.0] = 1.0
    return matrix / norms


class CodeIndex:
    """Holds chunks, their dense vectors, and a lexical index for hybrid search."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        matrix: np.ndarray,
        lexical: LexicalIndex,
    ) -> None:
        self.chunks = chunks
        self._embedder = embedder
        self.matrix = matrix
        self._lexical = lexical

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder | None = None) -> CodeIndex:
        resolved = embedder if embedder is not None else HashingEmbedder()
        vectors = resolved.embed([chunk.text for chunk in chunks])
        matrix = _as_normalized_matrix(vectors, resolved.dim)
        lexical = LexicalIndex(chunks)
        return cls(list(chunks), resolved, matrix, lexical)

    @classmethod
    def from_vectors(
        cls,
        chunks: list[Chunk],
        embedder: Embedder | str,
        vectors: np.ndarray | list[list[float]],
    ) -> CodeIndex:
        """Reconstruct an index from precomputed vectors, e.g. after loading.

        Accepts either an embedder instance or its name (resolved lazily) so
        store.py can serialize the matrix and rebuild without re-embedding.
        """
        resolved = resolve_embedder(embedder) if isinstance(embedder, str) else embedder
        rows = vectors.tolist() if isinstance(vectors, np.ndarray) else list(vectors)
        matrix = _as_normalized_matrix(rows, resolved.dim)
        lexical = LexicalIndex(chunks)
        return cls(list(chunks), resolved, matrix, lexical)

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    @property
    def dim(self) -> int:
        return self._embedder.dim

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def vectors(self) -> list[list[float]]:
        """Chunk vectors as plain nested lists, for serialization."""
        return self.matrix.tolist()

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 10, alpha: float = 0.5) -> list[SearchResult]:
        if not self.chunks or not query.strip():
            return []

        query_vector = np.asarray(self._embedder.embed([query])[0], dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if norm != 0.0:
            query_vector = query_vector / norm

        # Rows are already L2-normalized, so the dot product is cosine similarity.
        cosines = self.matrix @ query_vector

        vector_scores = {chunk.id: float(cosines[i]) for i, chunk in enumerate(self.chunks)}
        lexical_scores = self._lexical.scores(query)
        normalized_lexical = min_max_normalize(lexical_scores)
        fused = fuse(vector_scores, lexical_scores, alpha)

        top_k = min(k, len(self.chunks))
        ranked_ids = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:top_k]

        by_id = {chunk.id: chunk for chunk in self.chunks}
        results: list[SearchResult] = []
        for chunk_id in ranked_ids:
            results.append(
                SearchResult(
                    chunk=by_id[chunk_id],
                    score=fused[chunk_id],
                    vector_score=vector_scores.get(chunk_id, 0.0),
                    lexical_score=normalized_lexical.get(chunk_id, 0.0),
                )
            )
        return results
