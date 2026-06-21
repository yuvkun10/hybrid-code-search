"""Tests for the source-aware chunker."""

from __future__ import annotations

import os

from hybrid_code_search.chunker import (
    _chunk_id,
    chunk_paths,
    chunk_source,
    detect_language,
)
from hybrid_code_search.types import Chunk

_PY_SOURCE = '''"""Module docstring."""

import os


def alpha(x):
    return x + 1


def beta(y):
    return y * 2


class Widget:
    """A widget."""

    def render(self):
        return "widget"
'''


def test_python_chunks_have_expected_symbols_and_kinds() -> None:
    chunks = chunk_source(_PY_SOURCE, "sample.py")

    by_symbol = {(c.kind, c.symbol) for c in chunks}
    assert ("function", "alpha") in by_symbol
    assert ("function", "beta") in by_symbol
    assert ("class", "Widget") in by_symbol
    assert ("method", "Widget.render") in by_symbol


def test_python_chunk_line_ranges_are_one_based() -> None:
    chunks = chunk_source(_PY_SOURCE, "sample.py")
    by_symbol = {c.symbol: c for c in chunks if c.kind == "function"}

    src_lines = _PY_SOURCE.splitlines()

    alpha = by_symbol["alpha"]
    # 1-based, inclusive: the def line must be the start of the range.
    assert src_lines[alpha.start_line - 1].startswith("def alpha")
    assert alpha.start_line >= 1
    assert alpha.end_line >= alpha.start_line

    beta = by_symbol["beta"]
    assert src_lines[beta.start_line - 1].startswith("def beta")
    assert beta.start_line > alpha.end_line


def test_python_chunk_text_matches_line_range() -> None:
    chunks = chunk_source(_PY_SOURCE, "sample.py")
    src_lines = _PY_SOURCE.splitlines(keepends=True)

    for chunk in chunks:
        expected = "".join(src_lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.text == expected


def test_method_chunk_is_nested_within_class_range() -> None:
    chunks = chunk_source(_PY_SOURCE, "sample.py")
    cls = next(c for c in chunks if c.kind == "class")
    method = next(c for c in chunks if c.kind == "method")

    assert cls.start_line <= method.start_line
    assert method.end_line <= cls.end_line
    assert method.symbol == "Widget.render"


def test_all_chunks_are_chunk_instances() -> None:
    chunks = chunk_source(_PY_SOURCE, "sample.py")
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)


def test_unparseable_python_falls_back_to_block_chunks() -> None:
    broken = "def oops(:\n    this is not valid python\n" * 5
    chunks = chunk_source(broken, "broken.py")

    assert chunks
    assert all(c.kind == "block" for c in chunks)
    assert all(c.language == "python" for c in chunks)


def test_non_python_source_yields_block_chunks_without_raising() -> None:
    text = "line one\nline two\nline three\n"
    chunks = chunk_source(text, "notes.txt")

    assert chunks
    assert all(c.kind == "block" for c in chunks)
    assert chunks[0].language == "text"


def test_empty_source_returns_no_chunks() -> None:
    assert chunk_source("", "empty.txt") == []


def test_chunk_ids_are_stable_across_calls() -> None:
    first = chunk_source(_PY_SOURCE, "sample.py")
    second = chunk_source(_PY_SOURCE, "sample.py")

    assert [c.id for c in first] == [c.id for c in second]


def test_chunk_id_is_deterministic_for_same_inputs() -> None:
    assert _chunk_id("a.py", "function", "f", 1, 5) == _chunk_id("a.py", "function", "f", 1, 5)


def test_chunk_id_varies_with_path_and_range() -> None:
    base = _chunk_id("a.py", "function", "f", 1, 5)
    assert base != _chunk_id("b.py", "function", "f", 1, 5)
    assert base != _chunk_id("a.py", "class", "f", 1, 5)
    assert base != _chunk_id("a.py", "function", "g", 1, 5)
    assert base != _chunk_id("a.py", "function", "f", 2, 5)
    assert base != _chunk_id("a.py", "function", "f", 1, 6)


def test_detect_language_maps_known_extensions() -> None:
    assert detect_language("foo.py") == "python"
    assert detect_language("foo.js") == "javascript"
    assert detect_language("foo.ts") == "typescript"
    assert detect_language("foo.go") == "go"
    assert detect_language("foo.rs") == "rust"
    assert detect_language("foo.md") == "markdown"


def test_detect_language_is_case_insensitive() -> None:
    assert detect_language("FOO.PY") == "python"


def test_detect_language_defaults_to_text() -> None:
    assert detect_language("foo.unknown") == "text"
    assert detect_language("noextension") == "text"


def test_explicit_language_overrides_extension() -> None:
    text = "alpha\nbeta\ngamma\n"
    chunks = chunk_source(text, "data.py", language="text")
    assert all(c.kind == "block" for c in chunks)
    assert all(c.language == "text" for c in chunks)


def test_chunk_paths_skips_file_with_nul_byte(tmp_path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("hello\nworld\n", encoding="utf-8")
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"hello\x00world\n")

    chunks = chunk_paths([str(tmp_path)])

    paths = {os.path.realpath(c.path) for c in chunks}
    assert os.path.realpath(str(good)) in paths
    assert os.path.realpath(str(binary)) not in paths


def test_chunk_paths_skips_file_larger_than_max_file_bytes(tmp_path) -> None:
    small = tmp_path / "small.txt"
    small.write_text("tiny\n", encoding="utf-8")
    large = tmp_path / "large.txt"
    large.write_text("x\n" * 1000, encoding="utf-8")

    chunks = chunk_paths([str(tmp_path)], max_file_bytes=64)

    paths = {os.path.realpath(c.path) for c in chunks}
    assert os.path.realpath(str(small)) in paths
    assert os.path.realpath(str(large)) not in paths


def test_chunk_paths_does_not_descend_into_skipped_dirs(tmp_path) -> None:
    top = tmp_path / "kept.txt"
    top.write_text("kept\n", encoding="utf-8")
    skipped_dir = tmp_path / "node_modules"
    skipped_dir.mkdir()
    (skipped_dir / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    chunks = chunk_paths([str(tmp_path)])

    paths = {os.path.realpath(c.path) for c in chunks}
    assert os.path.realpath(str(top)) in paths
    assert all("node_modules" not in c.path for c in chunks)


def test_chunk_paths_deduplicates_same_file_passed_twice(tmp_path) -> None:
    target = tmp_path / "dup.py"
    target.write_text(_PY_SOURCE, encoding="utf-8")

    chunks = chunk_paths([str(target), str(target)])

    ids = [c.id for c in chunks]
    assert ids
    assert len(ids) == len(set(ids))
