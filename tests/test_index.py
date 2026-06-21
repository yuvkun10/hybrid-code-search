"""Tests for the in-memory hybrid :class:`CodeIndex`."""

from __future__ import annotations

from hybrid_code_search.chunker import chunk_source
from hybrid_code_search.embedder import HashingEmbedder
from hybrid_code_search.index import CodeIndex
from hybrid_code_search.types import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        path="example.py",
        language="python",
        kind="function",
        symbol=chunk_id,
        start_line=1,
        end_line=text.count("\n") + 1,
        text=text,
    )


def _sample_chunks() -> list[Chunk]:
    return [
        _chunk(
            "auth",
            "def authenticate_user(username, password):\n    return verify_password(password)\n",
        ),
        _chunk("parse", "def parse_json_payload(raw):\n    return json.loads(raw)\n"),
        _chunk(
            "render", "def render_html_template(context):\n    return template.render(context)\n"
        ),
        _chunk("matrix", "def multiply_matrix(left, right):\n    return left @ right\n"),
    ]


def test_matching_chunk_ranks_first_across_distinct_topics() -> None:
    index = CodeIndex.build(_sample_chunks())

    queries = {
        "authenticate user password": "auth",
        "parse json payload": "parse",
        "render html template": "render",
        "multiply matrix": "matrix",
    }
    for query, expected_id in queries.items():
        results = index.search(query)
        assert results
        assert results[0].chunk.id == expected_id
        for result in results:
            assert 0.0 <= result.score <= 1.0
            assert -1.0 <= result.vector_score <= 1.0
            assert 0.0 <= result.lexical_score <= 1.0


def test_chunk_source_query_ranks_matching_chunk_first() -> None:
    source = (
        "def authenticate_user(username, password):\n"
        "    return verify_password(password)\n"
        "\n"
        "\n"
        "def parse_json_payload(raw):\n"
        "    return json.loads(raw)\n"
        "\n"
        "\n"
        "def multiply_matrix(left, right):\n"
        "    return left @ right\n"
    )
    chunks = chunk_source(source, "example.py")
    index = CodeIndex.build(chunks)

    results = index.search("parse json payload")

    assert results
    assert results[0].chunk.symbol == "parse_json_payload"


def test_two_builds_rank_identically() -> None:
    chunks = _sample_chunks()

    first = CodeIndex.build(chunks).search("parse json payload")
    second = CodeIndex.build(chunks).search("parse json payload")

    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]
    assert [r.score for r in first] == [r.score for r in second]


def test_empty_index_returns_empty_list() -> None:
    index = CodeIndex.build([])

    assert index.search("anything") == []


def test_empty_query_returns_empty_list() -> None:
    index = CodeIndex.build(_sample_chunks())

    assert index.search("") == []
    assert index.search("   ") == []


def test_k_caps_number_of_results() -> None:
    index = CodeIndex.build(_sample_chunks())

    results = index.search("authenticate user", k=2)

    assert len(results) == 2


def test_k_larger_than_corpus_is_clamped() -> None:
    chunks = _sample_chunks()
    index = CodeIndex.build(chunks)

    results = index.search("authenticate user", k=100)

    assert len(results) == len(chunks)


def test_scores_are_within_expected_ranges() -> None:
    index = CodeIndex.build(_sample_chunks())

    results = index.search("authenticate user password")

    for result in results:
        assert 0.0 <= result.score <= 1.0
        assert -1.0 <= result.vector_score <= 1.0
        assert 0.0 <= result.lexical_score <= 1.0


def test_results_are_sorted_by_descending_score() -> None:
    index = CodeIndex.build(_sample_chunks())

    results = index.search("parse json payload")

    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_default_embedder_is_deterministic() -> None:
    chunks = _sample_chunks()

    first = CodeIndex.build(chunks).search("authenticate user password")
    second = CodeIndex.build(chunks).search("authenticate user password")

    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]
    assert [r.score for r in first] == [r.score for r in second]


def test_explicit_hashing_embedder_matches_default() -> None:
    chunks = _sample_chunks()

    default_results = CodeIndex.build(chunks).search("parse json payload")
    explicit_results = CodeIndex.build(chunks, HashingEmbedder()).search("parse json payload")

    assert [r.chunk.id for r in default_results] == [r.chunk.id for r in explicit_results]
    assert [r.score for r in default_results] == [r.score for r in explicit_results]
