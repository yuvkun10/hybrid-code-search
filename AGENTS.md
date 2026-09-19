# AGENTS.md

`hybrid-code-search` is a Python library, `scs` CLI and FastAPI service that ranks code chunks by fusing dense vector similarity with BM25.

## Setup

Python 3.11 or newer.

```bash
pip install -e ".[dev]"
pip install -e ".[transformers]"   # optional learned embedder
```

`SCS_EMBEDDER` and `SCS_MODEL` are needed only for the learned embedder. See [docs/configuration.md](docs/configuration.md).

## Commands

```bash
ruff check .
ruff format --check .
mypy src
pytest --cov=hybrid_code_search --cov-report=term-missing
scs index src/ --out code.index.json
scs search "open a file safely" --index-path code.index.json --k 10 --alpha 0.5
```

## Project structure

- `src/hybrid_code_search/chunker.py`, `tokenize.py`: AST aware chunking and tokens.
- `embedder.py`, `ranking.py`, `index.py`: embeddings, score fusion, index.
- `store.py`: JSON index store. `cli.py`: `scs` CLI. `service.py`: FastAPI app.
- `tests/`: pytest suites. `vault/`: design notes as an Obsidian vault.

Details are in [docs/architecture.md](docs/architecture.md).

## Conventions

- Ruff lints (rules `E`, `F`, `I`, `UP`, `B`, `SIM`, line length 100) and formats. mypy checks `src`. Settings live in `pyproject.toml`.
- The core (chunker, embedder, index, ranking) must not need network access or a model download. Anything that does belongs behind the `transformers` extra and must fail with a clear error when the extra is missing.
- New ranking or chunking behavior ships with a test that pins the expected ordering on a small, readable corpus.
- Do not add attribution trailers to commits.

## Testing

Before a PR run `ruff check . && ruff format --check . && mypy src && pytest --cov=hybrid_code_search`. CI runs the same checks.

## Safety

- Never commit `.env` files or generated `*.index.json` files.

## More

- [docs/README.md](docs/README.md): docs index
- [docs/usage.md](docs/usage.md): library, CLI and HTTP API
- [CONTRIBUTING.md](CONTRIBUTING.md): pull request checks
