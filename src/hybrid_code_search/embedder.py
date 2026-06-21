"""Text-to-vector embedders implementing the Embedder protocol."""

from __future__ import annotations

import hashlib
import os

import numpy as np

from .tokenize import tokenize
from .types import Embedder


class HashingEmbedder:
    """Deterministic feature-hashing embedder.

    Tokens are hashed into a fixed number of buckets with a signed count, then
    the vector is L2-normalized. Using blake2b rather than the builtin hash()
    keeps vectors stable across processes, since hash() is salted per run.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return "hashing"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float64)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self._dim
            # A separate bit decides the sign so collisions can cancel rather
            # than always reinforce, reducing hashing bias.
            sign = 1.0 if (value >> 1) & 1 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec.tolist()
        return (vec / norm).tolist()


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model.

    The heavy dependency is imported lazily so that importing this module does
    not require the optional "transformers" extra to be installed.
    """

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for "
                "SentenceTransformerEmbedder. Install the optional "
                "dependency with: pip install 'hybrid-code-search[transformers]'"
            ) from exc
        self._model_name = model
        self._model = SentenceTransformer(model)

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float64).tolist()


def resolve_embedder(name: str | None = None, model: str | None = None) -> Embedder:
    """Build an Embedder from explicit args or SCS_EMBEDDER/SCS_MODEL env vars."""
    selected = name or os.environ.get("SCS_EMBEDDER", "hashing")
    chosen_model = model or os.environ.get("SCS_MODEL")
    if selected == "sentence-transformers":
        if not chosen_model:
            raise ValueError(
                "A model name is required for the sentence-transformers "
                "embedder; set SCS_MODEL or pass model=..."
            )
        return SentenceTransformerEmbedder(chosen_model)
    return HashingEmbedder()
