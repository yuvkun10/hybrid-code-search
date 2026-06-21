# hybrid-code-search

Search a codebase by meaning, not just substrings. The engine splits source into
AST-aware chunks (functions, classes, methods), embeds them with a pluggable embedder,
and ranks results with a hybrid of dense vector similarity and lexical scoring so that
identifier matches and semantic matches both count.

The default embedder is deterministic and runs with no model download and no network,
so indexing and search work offline out of the box. A real transformer backend is an
optional extra.

Status: scaffolding. The chunker, index, and search land in the first feature PR.

## Why hybrid

Pure embeddings miss exact identifier and API-name matches that matter in code; pure
lexical search misses paraphrased intent. Fusing the two is the honest default for code.

## Install

    pip install -e ".[dev]"

## Development

    ruff check .
    ruff format --check .
    mypy src
    pytest --cov=hybrid_code_search

## License

MIT
