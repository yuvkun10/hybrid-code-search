"""Source-aware chunking of code and text files into :class:`Chunk` spans."""

from __future__ import annotations

import ast
import hashlib
import os
from collections.abc import Iterable

from .types import Chunk

_EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".md": "markdown",
}

_SKIP_DIRS: frozenset[str] = frozenset(
    {"node_modules", ".git", ".venv", "dist", "build", "__pycache__"}
)

_WINDOW_LINES = 40
_WINDOW_OVERLAP = 10
# A leading module chunk is only emitted when top-level code is non-trivial,
# so single-symbol files do not produce a redundant near-duplicate chunk.
_MIN_MODULE_LINES = 3


def detect_language(path: str) -> str:
    """Return the language name for ``path`` based on its file extension."""
    _, ext = os.path.splitext(path)
    return _EXTENSION_LANGUAGES.get(ext.lower(), "text")


def _chunk_id(path: str, kind: str, symbol: str, start_line: int, end_line: int) -> str:
    digest = hashlib.sha1(f"{path}:{kind}:{symbol}:{start_line}:{end_line}".encode())
    return digest.hexdigest()[:16]


def _make_chunk(
    *,
    path: str,
    language: str,
    kind: str,
    symbol: str,
    start_line: int,
    end_line: int,
    text: str,
) -> Chunk:
    return Chunk(
        id=_chunk_id(path, kind, symbol, start_line, end_line),
        path=path,
        language=language,
        kind=kind,
        symbol=symbol,
        start_line=start_line,
        end_line=end_line,
        text=text,
    )


def _slice_lines(lines: list[str], start_line: int, end_line: int) -> str:
    return "".join(lines[start_line - 1 : end_line])


def _sliding_window(text: str, path: str, language: str) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[Chunk] = []
    step = max(1, _WINDOW_LINES - _WINDOW_OVERLAP)
    total = len(lines)
    start = 0
    while start < total:
        end = min(start + _WINDOW_LINES, total)
        chunks.append(
            _make_chunk(
                path=path,
                language=language,
                kind="block",
                symbol="",
                start_line=start + 1,
                end_line=end,
                text="".join(lines[start:end]),
            )
        )
        if end >= total:
            break
        start += step
    return chunks


def _node_end_line(node: ast.AST, fallback: int) -> int:
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int):
        return end
    return fallback


def _chunk_python(text: str, path: str, language: str) -> list[Chunk]:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []

    def_types = (ast.FunctionDef, ast.AsyncFunctionDef)

    # The module chunk spans only the contiguous top-level statements before the
    # first def/class, capturing module docstrings, imports, and bare top-level
    # code without overlapping the function/class chunks that follow.
    first_def_start = None
    for stmt in tree.body:
        if isinstance(stmt, (*def_types, ast.ClassDef)):
            first_def_start = stmt.lineno
            break

    module_end = 0
    for stmt in tree.body:
        if isinstance(stmt, (*def_types, ast.ClassDef)):
            break
        module_end = max(module_end, _node_end_line(stmt, getattr(stmt, "lineno", 0)))

    if first_def_start is not None:
        module_end = min(module_end, first_def_start - 1)

    if module_end >= _MIN_MODULE_LINES:
        chunks.append(
            _make_chunk(
                path=path,
                language=language,
                kind="module",
                symbol="",
                start_line=1,
                end_line=module_end,
                text=_slice_lines(lines, 1, module_end),
            )
        )

    for node in tree.body:
        if isinstance(node, def_types):
            start = node.lineno
            end = _node_end_line(node, start)
            chunks.append(
                _make_chunk(
                    path=path,
                    language=language,
                    kind="function",
                    symbol=node.name,
                    start_line=start,
                    end_line=end,
                    text=_slice_lines(lines, start, end),
                )
            )
        elif isinstance(node, ast.ClassDef):
            start = node.lineno
            end = _node_end_line(node, start)
            chunks.append(
                _make_chunk(
                    path=path,
                    language=language,
                    kind="class",
                    symbol=node.name,
                    start_line=start,
                    end_line=end,
                    text=_slice_lines(lines, start, end),
                )
            )
            for member in node.body:
                if isinstance(member, def_types):
                    m_start = member.lineno
                    m_end = _node_end_line(member, m_start)
                    chunks.append(
                        _make_chunk(
                            path=path,
                            language=language,
                            kind="method",
                            symbol=f"{node.name}.{member.name}",
                            start_line=m_start,
                            end_line=m_end,
                            text=_slice_lines(lines, m_start, m_end),
                        )
                    )

    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def chunk_source(text: str, path: str, language: str | None = None) -> list[Chunk]:
    """Split ``text`` into chunks, using AST structure for Python source."""
    lang = language if language is not None else detect_language(path)
    if lang == "python":
        try:
            return _chunk_python(text, path, lang)
        except (SyntaxError, ValueError, RecursionError):
            return _sliding_window(text, path, lang)
    return _sliding_window(text, path, lang)


def chunk_file(path: str) -> list[Chunk]:
    """Read ``path`` as UTF-8 and chunk it; return ``[]`` if unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return []
    return chunk_source(text, path)


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            sample = handle.read(8192)
    except OSError:
        return True
    return b"\x00" in sample


def _eligible(path: str, max_file_bytes: int) -> bool:
    try:
        if os.path.getsize(path) > max_file_bytes:
            return False
    except OSError:
        return False
    return not _is_binary(path)


def _iter_files(path: str, max_file_bytes: int, seen: set[str]) -> Iterable[str]:
    if os.path.isfile(path):
        real = os.path.realpath(path)
        if real in seen:
            return
        if _eligible(path, max_file_bytes):
            seen.add(real)
            yield path
        return
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SKIP_DIRS]
        for name in files:
            file_path = os.path.join(root, name)
            real = os.path.realpath(file_path)
            if real in seen:
                continue
            if not _eligible(file_path, max_file_bytes):
                continue
            seen.add(real)
            yield file_path


def chunk_paths(paths: Iterable[str], *, max_file_bytes: int = 1_000_000) -> list[Chunk]:
    """Walk ``paths`` and aggregate chunks from every eligible file."""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for path in paths:
        for file_path in _iter_files(path, max_file_bytes, seen):
            chunks.extend(chunk_file(file_path))
    return chunks
