"""Command-line interface for hybrid-code-search."""

from __future__ import annotations

import typer

from .chunker import chunk_paths
from .embedder import resolve_embedder
from .index import CodeIndex
from .store import load_index, save_index

app = typer.Typer()

_SNIPPET_MAX = 120


@app.command()
def index(
    root: str,
    out: str = "code.index.json",
    embedder: str = "hashing",
) -> None:
    """Chunk a source tree and build a persisted hybrid index."""
    chunks = chunk_paths([root])
    code_index = CodeIndex.build(chunks, resolve_embedder(embedder))
    save_index(code_index, out)
    typer.echo(f"Indexed {len(chunks)} chunks -> {out}")


@app.command()
def search(
    query: str,
    index_path: str = "code.index.json",
    k: int = 10,
    alpha: float = 0.5,
) -> None:
    """Search a persisted index and print ranked results."""
    try:
        code_index = load_index(index_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Could not load index from {index_path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    results = code_index.search(query, k=k, alpha=alpha)
    for result in results:
        chunk = result.chunk
        location = f"{chunk.path}:{chunk.start_line}-{chunk.end_line}"
        symbol = chunk.symbol or chunk.kind
        typer.echo(f"{location} {symbol} {result.score:.4f}")
        snippet = " ".join(chunk.text.split())[:_SNIPPET_MAX]
        if snippet:
            typer.echo(f"    {snippet}")


if __name__ == "__main__":
    app()
