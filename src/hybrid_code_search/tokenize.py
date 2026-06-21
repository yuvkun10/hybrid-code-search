"""Lexical tokenization for code text."""

from __future__ import annotations

import re

# Matches runs of letters/digits, with internal boundaries handled separately.
# Splitting first on non-alphanumeric isolates identifier-like spans, which are
# then broken on camelCase and digit transitions.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Boundaries inside a single alphanumeric span:
#   acronym->word ("HTTPResponse" -> "HTTP", "Response")
#   word->word    ("parseHTTP"    -> "parse", "HTTP")
#   letter<->digit transitions
_CAMEL_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])"  # acronym before a capitalized word
    r"|[A-Z]?[a-z]+"  # a capitalized or lowercase word
    r"|[A-Z]+"  # a trailing run of capitals (acronym)
    r"|[0-9]+"  # a run of digits
)


def tokenize(text: str) -> list[str]:
    """Return lowercase lexical tokens for code text.

    Splits on non-alphanumeric characters and further decomposes camelCase,
    snake_case, and letter/digit boundaries. Single-character tokens are
    dropped unless purely numeric.
    """
    tokens: list[str] = []
    for span in _WORD_RE.findall(text):
        for part in _CAMEL_RE.findall(span):
            if len(part) == 1 and not part.isdigit():
                continue
            tokens.append(part.lower())
    return tokens
