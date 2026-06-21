"""Tests for BM25 lexical scoring, score normalization, and fusion."""

from __future__ import annotations

from hybrid_code_search.ranking import LexicalIndex, fuse, min_max_normalize
from hybrid_code_search.types import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        path=f"{chunk_id}.py",
        language="python",
        kind="function",
        symbol=chunk_id,
        start_line=1,
        end_line=1,
        text=text,
    )


def test_bm25_scores_chunk_with_query_term_above_chunk_without() -> None:
    chunks = [
        _chunk("with", "def parse_config(path): return load(path)"),
        _chunk("without", "def render_template(name): return draw(name)"),
    ]
    index = LexicalIndex(chunks)

    scores = index.scores("config")

    assert scores["with"] > scores["without"]
    assert scores["without"] == 0.0


def test_bm25_unmatched_query_scores_all_zero() -> None:
    chunks = [
        _chunk("a", "alpha beta gamma"),
        _chunk("b", "delta epsilon zeta"),
    ]
    index = LexicalIndex(chunks)

    scores = index.scores("nonexistent")

    assert set(scores) == {"a", "b"}
    assert all(value == 0.0 for value in scores.values())


def test_bm25_empty_query_scores_all_zero() -> None:
    index = LexicalIndex([_chunk("a", "alpha beta")])

    scores = index.scores("")

    assert scores == {"a": 0.0}


def test_bm25_empty_index_returns_empty_scores() -> None:
    index = LexicalIndex([])

    assert index.scores("anything") == {}


def test_min_max_normalize_empty() -> None:
    assert min_max_normalize({}) == {}


def test_min_max_normalize_all_equal_returns_zeros() -> None:
    result = min_max_normalize({"a": 3.0, "b": 3.0, "c": 3.0})

    assert result == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_min_max_normalize_scales_to_unit_range() -> None:
    result = min_max_normalize({"lo": 2.0, "mid": 4.0, "hi": 6.0})

    assert result["lo"] == 0.0
    assert result["hi"] == 1.0
    assert result["mid"] == 0.5


def test_fuse_alpha_one_normalizes_vector_only() -> None:
    # Un-normalized inputs: with alpha=1.0 the lexical side drops out, so the
    # result must equal the min-max-normalized vector. If fuse() skipped
    # normalization the result would be the raw {a: 5.0, b: 1.0}.
    vector = {"a": 5.0, "b": 1.0}
    lexical = {"a": 0.0, "b": 100.0}

    fused = fuse(vector, lexical, alpha=1.0)

    assert fused == {"a": 1.0, "b": 0.0}


def test_fuse_alpha_zero_normalizes_lexical_only() -> None:
    # Symmetric check on the lexical side with un-normalized inputs.
    vector = {"a": 100.0, "b": 0.0}
    lexical = {"a": 1.0, "b": 5.0}

    fused = fuse(vector, lexical, alpha=0.0)

    assert fused == {"a": 0.0, "b": 1.0}


def test_fuse_missing_ids_treated_as_zero() -> None:
    vector = {"a": 1.0, "b": 0.0}
    lexical = {"b": 1.0, "c": 0.0}

    fused = fuse(vector, lexical, alpha=0.5)

    # The union of ids is covered; ids missing from one side contribute 0 there.
    assert set(fused) == {"a", "b", "c"}
    # "a" is absent from lexical so its lexical component is the corpus minimum
    # (normalized to 0); its vector component is the max (1.0).
    assert fused["a"] == 0.5 * 1.0 + 0.5 * 0.0


def test_fuse_default_alpha_blends_normalized_sides_evenly() -> None:
    # Un-normalized but aligned inputs: each side normalizes to {a: 1.0, b: 0.0}
    # so the even blend lands on the same endpoints, proving both sides are
    # normalized before blending.
    vector = {"a": 5.0, "b": 1.0}
    lexical = {"a": 10.0, "b": 2.0}

    fused = fuse(vector, lexical)

    assert fused["a"] == 1.0
    assert fused["b"] == 0.0
