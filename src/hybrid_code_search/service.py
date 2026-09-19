"""HTTP service exposing the hybrid code search index."""

from __future__ import annotations

import os
from dataclasses import replace

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .chunker import _chunk_id, chunk_paths
from .embedder import resolve_embedder
from .index import CodeIndex
from .types import Chunk, SearchResult

_MAX_QUERY_CHARS = 2000


class IndexRequest(BaseModel):
    """Request body for building the index.

    Accepts either an explicit list of paths or a single root directory.
    """

    paths: list[str] | None = None
    root: str | None = None


class IndexResponse(BaseModel):
    chunks: int


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=1000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class ChunkModel(BaseModel):
    id: str
    path: str
    language: str
    kind: str
    symbol: str
    start_line: int
    end_line: int
    text: str

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkModel:
        return cls(
            id=chunk.id,
            path=chunk.path,
            language=chunk.language,
            kind=chunk.kind,
            symbol=chunk.symbol,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            text=chunk.text,
        )


class SearchResultModel(BaseModel):
    chunk: ChunkModel
    score: float
    vector_score: float
    lexical_score: float

    @classmethod
    def from_result(cls, result: SearchResult) -> SearchResultModel:
        return cls(
            chunk=ChunkModel.from_chunk(result.chunk),
            score=result.score,
            vector_score=result.vector_score,
            lexical_score=result.lexical_score,
        )


class HealthResponse(BaseModel):
    status: str
    version: str
    chunks: int


def _resolve_within(base: str, requested: str) -> str:
    """Resolve ``requested`` against ``base`` and reject anything outside it."""
    try:
        candidate = os.path.realpath(os.path.join(base, requested))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc
    if candidate == base:
        return base
    if not candidate.startswith(os.path.join(base, "")):
        raise HTTPException(status_code=400, detail="path is outside the allowed root")
    return candidate


def _relative_to(base: str, chunk: Chunk) -> Chunk:
    """Report chunk paths relative to ``base`` so responses never expose server paths."""
    rel = os.path.relpath(chunk.path, base)
    return replace(
        chunk,
        path=rel,
        id=_chunk_id(rel, chunk.kind, chunk.symbol, chunk.start_line, chunk.end_line),
    )


def create_app(index: CodeIndex | None = None, *, root: str | None = None) -> FastAPI:
    """Build a FastAPI app holding a mutable, optionally pre-populated index.

    ``POST /index`` only reads paths inside ``root``, which defaults to the
    current working directory.
    """
    app = FastAPI(title="hybrid-code-search", version=__version__)
    allowed_root = os.path.realpath(root if root is not None else os.getcwd())

    # The index is held on app.state so routes and re-indexing can mutate it
    # without rebinding a module-level global, which keeps the app testable.
    app.state.index = index if index is not None else CodeIndex.build([], resolve_embedder())

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current: CodeIndex = app.state.index
        return HealthResponse(
            status="ok",
            version=__version__,
            chunks=len(current),
        )

    @app.post("/index", response_model=IndexResponse)
    def build_index(request: IndexRequest) -> IndexResponse:
        provided_paths = [p for p in (request.paths or []) if p]
        if provided_paths:
            paths = provided_paths
        elif request.root:
            paths = [request.root]
        else:
            raise HTTPException(status_code=400, detail="paths or root is required")

        resolved = [_resolve_within(allowed_root, p) for p in paths]
        chunks = [_relative_to(allowed_root, chunk) for chunk in chunk_paths(resolved)]
        new_index = CodeIndex.build(chunks, resolve_embedder())
        app.state.index = new_index
        return IndexResponse(chunks=len(new_index))

    @app.post("/search", response_model=list[SearchResultModel])
    def search(request: SearchRequest) -> list[SearchResultModel]:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query must not be empty")
        if len(query) > _MAX_QUERY_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"query exceeds {_MAX_QUERY_CHARS} characters",
            )

        current: CodeIndex = app.state.index
        results = current.search(query, k=request.k, alpha=request.alpha)
        return [SearchResultModel.from_result(r) for r in results]

    return app
