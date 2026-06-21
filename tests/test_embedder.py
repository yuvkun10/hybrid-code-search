"""Tests for the deterministic HashingEmbedder."""

from __future__ import annotations

import math

import numpy as np

from hybrid_code_search.embedder import HashingEmbedder


def test_determinism_across_instances() -> None:
    text = "def parseHTTPResponse(request): return request.body"
    first = HashingEmbedder().embed([text])[0]
    second = HashingEmbedder().embed([text])[0]
    assert first == second


def test_dimension_equals_dim() -> None:
    embedder = HashingEmbedder(dim=128)
    [vector] = embedder.embed(["some sample identifier tokens"])
    assert embedder.dim == 128
    assert len(vector) == 128


def test_vectors_are_l2_normalized() -> None:
    embedder = HashingEmbedder()
    [vector] = embedder.embed(["compute the cosine similarity of two vectors"])
    norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))
    assert math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_empty_token_set_yields_zero_vector() -> None:
    embedder = HashingEmbedder(dim=64)
    # "" produces no tokens, and "a" is a single non-numeric token that the
    # tokenizer drops, so both yield an all-zeros (non-normalized) vector.
    for text in ("", "a", "   ", "!!!"):
        [vector] = embedder.embed([text])
        assert len(vector) == 64
        assert all(component == 0.0 for component in vector)


def test_different_texts_give_different_vectors() -> None:
    embedder = HashingEmbedder()
    [first] = embedder.embed(["the quick brown fox"])
    [second] = embedder.embed(["lazy dogs sleep soundly"])
    assert first != second
