---
title: hybrid-code-search vault
---

This folder is an Obsidian vault for the `hybrid-code-search` project documentation. Open it as a vault in Obsidian (File > Open vault, then select this `vault/` directory) and start at [[Home]]. The notes use wikilinks, so navigation works best inside Obsidian; reading the raw Markdown on disk also works but cross-links will not resolve.

## Contents

- [[Home]] — entry point and orientation.

## Scope

The notes describe the package under `src/hybrid_code_search` as it exists in the source tree: the chunker, tokenizer, embedders, lexical (BM25) index, score fusion, the in-memory `CodeIndex`, JSON persistence, the `scs` CLI, and the FastAPI service module. Where the code and docs disagree, the source is authoritative.
