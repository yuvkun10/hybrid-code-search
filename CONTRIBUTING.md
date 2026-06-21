# Contributing

Before opening a PR:

    pip install -e ".[dev]"
    ruff check . && ruff format --check . && mypy src && pytest --cov=hybrid_code_search

The core (chunker, embedder, index, ranking) must not require network access or a model
download. Anything that does belongs behind the optional `transformers` extra and must
degrade with a clear error when the extra is not installed.

New ranking or chunking behavior should ship with a test that pins the expected ordering
on a small, readable corpus.
