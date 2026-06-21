"""End-to-end CLI tests for the index and search commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hybrid_code_search.cli import app

runner = CliRunner()

_SAMPLE_SOURCE = '''\
"""A tiny sample module used by the CLI tests."""


def compute_widget_total(items: list[int]) -> int:
    """Sum the widget quantities."""
    total = 0
    for value in items:
        total += value
    return total
'''


def _write_sample(tmp_path: Path) -> Path:
    source = tmp_path / "sample.py"
    source.write_text(_SAMPLE_SOURCE, encoding="utf-8")
    return source


def test_index_then_search_finds_chunk(tmp_path: Path) -> None:
    source = _write_sample(tmp_path)
    index_path = tmp_path / "code.index.json"

    index_result = runner.invoke(app, ["index", str(source), "--out", str(index_path)])
    assert index_result.exit_code == 0, index_result.output
    assert index_path.exists()

    search_result = runner.invoke(
        app, ["search", "compute_widget_total", "--index-path", str(index_path)]
    )
    assert search_result.exit_code == 0, search_result.output
    assert str(source) in search_result.output


def test_search_missing_index_exits_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.index.json"

    result = runner.invoke(app, ["search", "anything", "--index-path", str(missing)])
    assert result.exit_code != 0
