# hybrid-code-search

Search a codebase by meaning and by name. The engine splits source into AST-aware
chunks, embeds each chunk into a dense vector, and ranks results by fusing cosine
similarity with a BM25 lexical score. The default embedder is deterministic feature
hashing, so indexing and search run offline with no model download. Version 0.1.0,
usable as a Python library, the `scs` CLI, or a FastAPI service.

## Installation

Requires Python 3.11 or newer.

```bash
pip install -e ".[dev]"
```

The optional learned-embedding backend:

```bash
pip install -e ".[transformers]"
```

Environment variables, only needed for that backend: `SCS_EMBEDDER`, `SCS_MODEL`.
See [docs/configuration.md](docs/configuration.md).

## Usage

Build an index of a source tree and search it:

```bash
scs index src/ --out code.index.json
scs search "open a file safely" --index-path code.index.json --k 10 --alpha 0.5
```

From Python:

```python
from hybrid_code_search import build_index

index = build_index(["src/"])
results = index.search("parse json config", k=5, alpha=0.5)
```

The HTTP service is created with `hybrid_code_search.service.create_app` and run with
uvicorn. The library, CLI and HTTP API are covered in [docs/usage.md](docs/usage.md).
There is no deployment setup in this repository.

## Project structure

```text
├── src
│   └── hybrid_code_search
│       ├── chunker.py
│       ├── embedder.py
│       ├── tokenize.py
│       ├── ranking.py
│       ├── index.py
│       ├── store.py
│       ├── cli.py
│       └── service.py
├── tests
├── docs
│   ├── architecture.md
│   ├── diagrams
│   └── archive
├── vault
├── CONTRIBUTING.md
└── pyproject.toml
```

How the pieces fit together: [docs/architecture.md](docs/architecture.md).

## Coding style

Ruff lints (rules `E`, `F`, `I`, `UP`, `B`, `SIM`, line length 100) and formats the
code, and mypy type checks `src`. Settings live in `pyproject.toml` and CI runs all
three on every pull request:

```bash
ruff check .
ruff format --check .
mypy src
```

## Test

```bash
pytest --cov=hybrid_code_search
```

The pytest suite in `tests/` covers the chunker, tokenizer, embedders, ranking,
index, JSON store, CLI and HTTP service, plus a smoke test.

## Documentation

- [docs/README.md](docs/README.md): index of all docs
- [docs/architecture.md](docs/architecture.md): pipeline, modules and limitations
- [docs/usage.md](docs/usage.md): library, CLI and HTTP API reference
- [docs/configuration.md](docs/configuration.md): embedder and ranking settings
- [CONTRIBUTING.md](CONTRIBUTING.md): pull request checks
- [vault/](vault/README.md): design notes as an Obsidian vault

## License

MIT. See [LICENSE](LICENSE).
