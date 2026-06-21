"""Persistence for a built :class:`CodeIndex`.

The on-disk format is JSON so an index can be inspected and diffed. Vectors are
stored verbatim so that loading never re-embeds: this keeps load deterministic
and independent of whether the original (possibly heavy) embedder backend is
available at load time.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from typing import Any

from .embedder import HashingEmbedder, resolve_embedder
from .index import CodeIndex
from .types import Chunk, Embedder

_VERSION = 1

_CHUNK_FIELDS = tuple(f.name for f in dataclasses.fields(Chunk))


def save_index(index: CodeIndex, path: str) -> None:
    """Serialize ``index`` to ``path`` as JSON.

    The write goes to a temporary sibling file that is then renamed over the
    target, so a crash mid-write cannot leave a half-written index in place.
    """
    payload: dict[str, Any] = {
        "version": _VERSION,
        "embedder": {
            "name": index.embedder.name,
            "dim": index.embedder.dim,
        },
        "chunks": [dataclasses.asdict(chunk) for chunk in index.chunks],
        "vectors": index.vectors,
    }

    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        os.remove(tmp_path)
        raise
    os.replace(tmp_path, path)


def load_index(path: str, embedder: Embedder | None = None) -> CodeIndex:
    """Reconstruct a :class:`CodeIndex` from JSON written by :func:`save_index`.

    Stored vectors are reused as-is; the lexical index is rebuilt from the
    chunks. ``embedder`` is only used to answer future queries and to validate
    dimensionality; it is never used to re-embed the stored chunks. When it is
    omitted, the embedder named in the file is resolved, falling back to a
    hashing embedder of the stored dimension when the named backend is unknown.
    """
    with open(path, encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Index file is not valid JSON: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Index file must contain a JSON object")

    version = raw.get("version")
    if version != _VERSION:
        raise ValueError(f"Unsupported index version: {version!r}")

    embedder_meta = raw.get("embedder")
    if not isinstance(embedder_meta, dict):
        raise ValueError("Index file is missing a valid 'embedder' section")
    name = embedder_meta.get("name")
    dim = embedder_meta.get("dim")
    if not isinstance(name, str) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("Index file has malformed embedder metadata")

    chunks = _parse_chunks(raw.get("chunks"))
    vectors = _parse_vectors(raw.get("vectors"), expected_dim=dim)

    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
        )

    resolved = embedder if embedder is not None else _resolve(name, dim)
    if resolved.dim != dim:
        raise ValueError(f"Embedder dimension {resolved.dim} does not match stored dimension {dim}")

    return CodeIndex.from_vectors(chunks, resolved, vectors)


def _resolve(name: str, dim: int) -> Embedder:
    """Resolve the stored embedder name, falling back to a hashing embedder.

    Loading must not fail just because the originally configured backend is no
    longer installable; the stored vectors are authoritative for ranking.
    """
    if name == "hashing":
        return HashingEmbedder(dim=dim)
    try:
        return resolve_embedder(name)
    except (ValueError, KeyError, ImportError):
        return HashingEmbedder(dim=dim)


def _parse_chunks(raw_chunks: Any) -> list[Chunk]:
    if not isinstance(raw_chunks, list):
        raise ValueError("Index file is missing a valid 'chunks' list")

    chunks: list[Chunk] = []
    for position, item in enumerate(raw_chunks):
        if not isinstance(item, dict):
            raise ValueError(f"Chunk at position {position} is not an object")
        missing = [field for field in _CHUNK_FIELDS if field not in item]
        if missing:
            raise ValueError(f"Chunk at position {position} is missing fields: {missing}")
        try:
            chunks.append(Chunk(**{field: item[field] for field in _CHUNK_FIELDS}))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Chunk at position {position} has invalid fields: {exc}") from exc
    return chunks


def _parse_vectors(raw_vectors: Any, expected_dim: int) -> list[list[float]]:
    if not isinstance(raw_vectors, list):
        raise ValueError("Index file is missing a valid 'vectors' list")

    vectors: list[list[float]] = []
    for position, item in enumerate(raw_vectors):
        if not isinstance(item, list):
            raise ValueError(f"Vector at position {position} is not a list")
        if len(item) != expected_dim:
            raise ValueError(
                f"Vector at position {position} has length {len(item)}, expected {expected_dim}"
            )
        try:
            vectors.append([float(value) for value in item])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Vector at position {position} has non-numeric values: {exc}"
            ) from exc
    return vectors
