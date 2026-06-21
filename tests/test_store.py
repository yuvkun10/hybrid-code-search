from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_code_search import CodeIndex, HashingEmbedder, load_index, save_index
from hybrid_code_search.types import Chunk


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="a",
            path="src/auth.py",
            language="python",
            kind="function",
            symbol="login",
            start_line=1,
            end_line=4,
            text="def login(user, password):\n    return authenticate(user, password)",
        ),
        Chunk(
            id="b",
            path="src/math.py",
            language="python",
            kind="function",
            symbol="add",
            start_line=1,
            end_line=2,
            text="def add(x, y):\n    return x + y",
        ),
        Chunk(
            id="c",
            path="src/db.py",
            language="python",
            kind="function",
            symbol="connect_database",
            start_line=1,
            end_line=3,
            text="def connect_database(url):\n    return Connection(url)",
        ),
    ]


def _build_index() -> CodeIndex:
    return CodeIndex.build(_make_chunks(), HashingEmbedder())


def test_round_trip_preserves_top_result(tmp_path: Path) -> None:
    original = _build_index()
    query = "authenticate user login password"
    expected = original.search(query, k=3)
    assert expected, "fixture query should match at least one chunk"

    path = str(tmp_path / "index.json")
    save_index(original, path)
    loaded = load_index(path)

    assert loaded.dim == original.dim
    assert loaded.embedder_name == original.embedder_name

    actual = loaded.search(query, k=3)
    assert actual
    assert actual[0].chunk.id == expected[0].chunk.id


def test_round_trip_reuses_stored_vectors_without_drift(tmp_path: Path) -> None:
    original = _build_index()
    query = "connect database url"
    expected = original.search(query, k=3)

    path = str(tmp_path / "index.json")
    save_index(original, path)
    loaded = load_index(path)
    actual = loaded.search(query, k=3)

    assert [r.chunk.id for r in actual] == [r.chunk.id for r in expected]
    for got, want in zip(actual, expected, strict=True):
        assert got.vector_score == pytest.approx(want.vector_score)
        assert got.score == pytest.approx(want.score)


def test_round_trip_preserves_chunk_payload(tmp_path: Path) -> None:
    original = _build_index()
    path = str(tmp_path / "index.json")
    save_index(original, path)
    loaded = load_index(path)

    assert loaded.chunks == original.chunks


def test_save_then_load_explicit_embedder(tmp_path: Path) -> None:
    original = _build_index()
    path = str(tmp_path / "index.json")
    save_index(original, path)

    loaded = load_index(path, embedder=HashingEmbedder())
    query = "authenticate user login password"
    assert loaded.search(query, k=1)[0].chunk.id == "a"


def test_load_with_mismatched_embedder_dim_raises_value_error(tmp_path: Path) -> None:
    original = _build_index()
    path = str(tmp_path / "index.json")
    save_index(original, path)

    mismatched = HashingEmbedder(dim=original.dim + 1)
    with pytest.raises(ValueError):
        load_index(path, embedder=mismatched)


def test_malformed_json_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_index(str(path))


def test_non_object_json_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_index(str(path))


def test_unsupported_version_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text('{"version": 999, "chunks": [], "vectors": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_index(str(path))


def test_chunk_vector_count_mismatch_raises_value_error(tmp_path: Path) -> None:
    original = _build_index()
    path = str(tmp_path / "index.json")
    save_index(original, path)

    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["vectors"] = data["vectors"][:-1]
    Path(path).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_index(path)
