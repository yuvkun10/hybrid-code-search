"""Lexical scoring (BM25) and score fusion for hybrid retrieval."""

from __future__ import annotations

import math
from collections import Counter

from .tokenize import tokenize
from .types import Chunk

_BM25_K1 = 1.5
_BM25_B = 0.75


class LexicalIndex:
    """BM25 index over the tokenized text of a chunk collection.

    Document frequencies, term frequencies, and lengths are computed once at
    construction so that scoring is a pure lookup and remains deterministic.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._ids: list[str] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_lengths: dict[str, int] = {}
        doc_freqs: Counter[str] = Counter()

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tf: Counter[str] = Counter(tokens)
            self._ids.append(chunk.id)
            self._term_freqs.append(tf)
            self._doc_lengths[chunk.id] = len(tokens)
            doc_freqs.update(tf.keys())

        self._doc_count = len(self._ids)
        # Average over the corpus; guard the empty case to avoid division by zero.
        total_length = sum(self._doc_lengths.values())
        self._avg_length = total_length / self._doc_count if self._doc_count else 0.0

        # Precompute IDF per term to keep scoring cheap.
        self._idf: dict[str, float] = {}
        for term, df in doc_freqs.items():
            self._idf[term] = math.log(1.0 + (self._doc_count - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[str, float]:
        """Return chunk.id -> raw BM25 score; unmatched chunks score 0."""
        query_terms = set(tokenize(query))
        result: dict[str, float] = {chunk_id: 0.0 for chunk_id in self._ids}
        if not query_terms or self._avg_length == 0.0:
            return result

        for chunk_id, tf in zip(self._ids, self._term_freqs, strict=True):
            length = self._doc_lengths[chunk_id]
            norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * length / self._avg_length)
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                score += idf * (freq * (_BM25_K1 + 1.0)) / (freq + norm)
            result[chunk_id] = score
        return result


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Scale values into [0, 1]; return zeros when all values are equal."""
    if not scores:
        return {}
    values = scores.values()
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0.0:
        return dict.fromkeys(scores, 0.0)
    return {key: (value - lo) / span for key, value in scores.items()}


def fuse(
    vector_scores: dict[str, float],
    lexical_scores: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    """Combine min-max normalized vector and lexical scores over their union."""
    ids = set(vector_scores) | set(lexical_scores)
    vector_aligned = {chunk_id: vector_scores.get(chunk_id, 0.0) for chunk_id in ids}
    lexical_aligned = {chunk_id: lexical_scores.get(chunk_id, 0.0) for chunk_id in ids}

    vector_norm = min_max_normalize(vector_aligned)
    lexical_norm = min_max_normalize(lexical_aligned)

    return {
        chunk_id: alpha * vector_norm[chunk_id] + (1.0 - alpha) * lexical_norm[chunk_id]
        for chunk_id in ids
    }
