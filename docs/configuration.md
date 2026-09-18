# Configuration

## Embedder

Selected via constructor argument, the `--embedder` CLI flag, or the
`resolve_embedder` helper, which reads environment variables when no argument is given:

- `SCS_EMBEDDER`: `hashing` (default) or `sentence-transformers`.
- `SCS_MODEL`: model name, required when `SCS_EMBEDDER=sentence-transformers`.

The `sentence-transformers` backend requires the `transformers` extra; constructing it
without the dependency installed raises a clear `ImportError`. Selecting it without a
model name raises `ValueError`. See [`.env.example`](../.env.example).

## alpha

The fusion weight in `[0, 1]`. `alpha = 1.0` is pure vector similarity, `alpha = 0.0`
is pure BM25, `alpha = 0.5` (default) weights them equally. Raise it for conceptual
queries, lower it for exact-identifier queries.

## k

Maximum number of results to return. The library clamps it to the number of chunks;
the HTTP API constrains it to `[1, 1000]`.

## Chunking

`chunk_paths(paths, max_file_bytes=1_000_000)` controls the maximum file size to
consider. Skip directories and window size (40 lines, 10-line overlap) are fixed
constants in `chunker.py`.
