# Usage reference

How to use `hybrid-code-search` as a library, from the `scs` CLI, and over HTTP.

## Library

`build_index` chunks one or more paths and returns a queryable `CodeIndex`. With no
embedder argument it resolves one from the environment (default: the hashing embedder).

```python
from hybrid_code_search import build_index

index = build_index(["src/"])
results = index.search("parse json config", k=5, alpha=0.5)

for r in results:
    c = r.chunk
    print(f"{c.path}:{c.start_line}-{c.end_line} {c.symbol or c.kind} {r.score:.4f}")
```

Each `SearchResult` carries the matched `Chunk`, the fused `score` (in `[0, 1]`), the
raw `vector_score` (cosine similarity in `[-1, 1]`), and the min-max normalized
`lexical_score` (in `[0, 1]`).

To build an index directly from `Chunk` objects, or with an explicit embedder:

```python
from hybrid_code_search import CodeIndex, HashingEmbedder, chunk_paths

chunks = chunk_paths(["src/"])
index = CodeIndex.build(chunks, HashingEmbedder(dim=256))
results = index.search("retry with backoff", k=10, alpha=0.7)
```

`CodeIndex.build(chunks, embedder=None)` falls back to a default `HashingEmbedder` when
no embedder is passed. `CodeIndex.search(query, k=10, alpha=0.5)` returns an empty list
for an empty index or a blank query. `k` is clamped to the number of chunks.

The index exposes `.chunks`, `.matrix`, `.vectors`, `.embedder`, `.embedder_name`,
`.dim`, and `len(index)`.

### Persistence

The store module serializes an index to JSON, including the stored vectors, so loading
never re-embeds:

```python
from hybrid_code_search import save_index, load_index

save_index(index, "code.index.json")
index = load_index("code.index.json")
```

## CLI

The package installs a `scs` command (two subcommands).

Build a persisted index from a source tree:

```bash
scs index src/ --out code.index.json --embedder hashing
```

Search a persisted index:

```bash
scs search "open a file safely" --index-path code.index.json --k 10 --alpha 0.5
```

`scs index` takes the source tree as a positional argument; defaults: `--out
code.index.json`, `--embedder hashing`. `scs search` takes the query as a positional
argument; defaults: `--index-path code.index.json`, `--k 10`, `--alpha 0.5`. Search
prints one line per result (`path:start-end symbol score`) followed by an indented,
whitespace-collapsed snippet capped at 120 characters. If the index file is missing or
unreadable, `scs search` prints an error and exits with status 1.

## HTTP API

The service is a FastAPI app created by `create_app`. It holds a single in-memory index
on application state; `POST /index` rebuilds it. `create_app(index=None)` optionally
accepts a pre-built index; otherwise it starts with an empty one.

```python
import uvicorn
from hybrid_code_search.service import create_app

uvicorn.run(create_app(), host="127.0.0.1", port=8000)
```

### `POST /index`

Rebuilds the index from disk. Provide either `paths` (a list) or `root` (a single
directory); one is required, or the call returns 400.

```json
{"root": "src/"}
{"paths": ["src/", "lib/"]}
```

Response:

```json
{"chunks": 128}
```

### `POST /search`

```json
{"query": "decode base64", "k": 10, "alpha": 0.5}
```

`k` is constrained to `[1, 1000]`, `alpha` to `[0.0, 1.0]`. An empty query or one over
2000 characters returns 400. Response is a list of search results:

```json
[
  {
    "chunk": {
      "id": "…", "path": "src/x.py", "language": "python",
      "kind": "function", "symbol": "decode", "start_line": 10,
      "end_line": 24, "text": "…"
    },
    "score": 0.81, "vector_score": 0.42, "lexical_score": 1.0
  }
]
```

### `GET /health`

```json
{"status": "ok", "version": "0.1.0", "chunks": 128}
```
