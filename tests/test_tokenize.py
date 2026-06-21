from __future__ import annotations

import pytest

from hybrid_code_search.tokenize import tokenize


def test_empty_string_returns_empty_list() -> None:
    assert tokenize("") == []


def test_whitespace_only_returns_empty_list() -> None:
    assert tokenize("   \t\n  ") == []


def test_lowercases_tokens() -> None:
    assert tokenize("HELLO World") == ["hello", "world"]


def test_camel_case_split() -> None:
    assert tokenize("parseValue") == ["parse", "value"]


def test_camel_case_acronym_boundary() -> None:
    assert tokenize("HTTPResponse") == ["http", "response"]


def test_camel_case_word_then_acronym() -> None:
    assert tokenize("parseHTTP") == ["parse", "http"]


def test_snake_case_split() -> None:
    assert tokenize("parse_http_value") == ["parse", "http", "value"]


def test_pascal_case_split() -> None:
    assert tokenize("MyClassName") == ["my", "class", "name"]


def test_letter_digit_transitions_split() -> None:
    assert tokenize("foo123bar") == ["foo", "123", "bar"]


def test_punctuation_is_a_separator() -> None:
    assert tokenize("foo.bar(baz)") == ["foo", "bar", "baz"]


def test_mixed_punctuation_and_whitespace() -> None:
    # Single-character parts are dropped, so "a_b-c d,e;f" yields nothing.
    assert tokenize("a_b-c d,e;f") == []
    assert tokenize("load->next") == ["load", "next"]


def test_single_alpha_char_tokens_dropped() -> None:
    # Single non-numeric parts are dropped; "x" alone yields nothing.
    assert tokenize("x") == []


def test_single_digit_token_kept() -> None:
    assert tokenize("9") == ["9"]


def test_single_char_dropped_in_camel_context() -> None:
    # The leading "a" is a single-letter part and is dropped.
    assert tokenize("aValue") == ["value"]


def test_returns_list_of_str() -> None:
    result = tokenize("snake_case andCamel")
    assert isinstance(result, list)
    assert all(isinstance(tok, str) for tok in result)


def test_deterministic() -> None:
    text = "getHTTPResponseCode_v2"
    assert tokenize(text) == tokenize(text)


def test_only_punctuation_returns_empty_list() -> None:
    assert tokenize("...---___") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("snake_case_name", ["snake", "case", "name"]),
        ("camelCaseName", ["camel", "case", "name"]),
        ("XMLHttpRequest", ["xml", "http", "request"]),
        ("value2Pixels", ["value", "2", "pixels"]),
    ],
)
def test_splitting_table(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected
