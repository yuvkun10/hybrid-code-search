"""End-to-end tests for the FastAPI service over a hermetic temporary index."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hybrid_code_search.service import create_app

_SAMPLE_SOURCE = '''\
"""A small sample module used to exercise the index."""

import math


def compute_average(values: list[float]) -> float:
    """Return the arithmetic mean of the given values."""
    if not values:
        return 0.0
    return math.fsum(values) / len(values)


class Calculator:
    """A tiny calculator holding a running total."""

    def __init__(self) -> None:
        self.total = 0.0

    def add(self, amount: float) -> float:
        """Add an amount to the running total and return it."""
        self.total += amount
        return self.total
'''

_TIMESTAMP_SOURCE = '''\
"""Date and time parsing helpers."""


def parse_iso_timestamp(text: str) -> tuple[int, int, int]:
    """Parse an ISO timestamp string into year, month, and day integers."""
    year, month, day = text.split("-")
    return int(year), int(month), int(day)
'''

_LIST_SOURCE = '''\
"""Linked list manipulation helpers."""


def reverse_linked_list(head: object) -> object:
    """Reverse a singly linked list and return the new head node."""
    previous = None
    current = head
    while current is not None:
        previous, current = current, getattr(current, "next", None)
    return previous
'''


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    source_file = tmp_path / "sample.py"
    source_file.write_text(_SAMPLE_SOURCE, encoding="utf-8")
    return TestClient(create_app(root=str(tmp_path)))


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 0


def test_index_returns_positive_chunk_count(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/index", json={"paths": [str(tmp_path)]})
    assert response.status_code == 200
    assert response.json()["chunks"] > 0


def test_index_then_health_reports_chunks(client: TestClient, tmp_path: Path) -> None:
    indexed = client.post("/index", json={"paths": [str(tmp_path)]}).json()["chunks"]
    health = client.get("/health").json()
    assert health["chunks"] == indexed


def test_search_returns_ranked_results(client: TestClient, tmp_path: Path) -> None:
    client.post("/index", json={"paths": [str(tmp_path)]})

    response = client.post("/search", json={"query": "compute average of values"})
    assert response.status_code == 200

    results = response.json()
    assert isinstance(results, list)
    assert len(results) > 0

    for item in results:
        assert item["chunk"]["path"]
        assert "symbol" in item["chunk"]
        assert isinstance(item["score"], float)
        assert isinstance(item["vector_score"], float)
        assert isinstance(item["lexical_score"], float)

    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_result_matches_query(tmp_path: Path) -> None:
    (tmp_path / "timestamps.py").write_text(_TIMESTAMP_SOURCE, encoding="utf-8")
    (tmp_path / "lists.py").write_text(_LIST_SOURCE, encoding="utf-8")

    app = create_app(root=str(tmp_path))
    client = TestClient(app)
    client.post("/index", json={"paths": [str(tmp_path)]})

    response = client.post("/search", json={"query": "parse iso timestamp string"})
    assert response.status_code == 200

    results = response.json()
    assert len(results) > 0
    assert results[0]["chunk"]["symbol"] == "parse_iso_timestamp"
    assert results[0]["chunk"]["path"].endswith("timestamps.py")


def test_search_empty_query_returns_400(client: TestClient, tmp_path: Path) -> None:
    client.post("/index", json={"paths": [str(tmp_path)]})

    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 400


def test_index_requires_paths_or_root(client: TestClient) -> None:
    response = client.post("/index", json={})
    assert response.status_code == 400


def test_index_empty_paths_returns_400(client: TestClient) -> None:
    response = client.post("/index", json={"paths": []})
    assert response.status_code == 400


def test_index_accepts_relative_path_inside_root(client: TestClient) -> None:
    response = client.post("/index", json={"root": "sample.py"})
    assert response.status_code == 200
    assert response.json()["chunks"] > 0


@pytest.mark.parametrize("escape", ["..", "../..", "/", "sample.py/../../outside"])
def test_index_rejects_paths_outside_root(client: TestClient, escape: str) -> None:
    response = client.post("/index", json={"paths": [escape]})
    assert response.status_code == 400
    assert client.get("/health").json()["chunks"] == 0


def test_index_rejects_sibling_with_shared_prefix(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sibling = tmp_path / "repo-secrets"
    root.mkdir()
    sibling.mkdir()
    (sibling / "leak.py").write_text(_SAMPLE_SOURCE, encoding="utf-8")

    client = TestClient(create_app(root=str(root)))
    response = client.post("/index", json={"root": str(sibling)})
    assert response.status_code == 400


def test_index_rejects_symlink_escaping_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "leak.py").write_text(_SAMPLE_SOURCE, encoding="utf-8")
    (root / "link").symlink_to(outside, target_is_directory=True)

    client = TestClient(create_app(root=str(root)))
    response = client.post("/index", json={"paths": ["link"]})
    assert response.status_code == 400


def test_index_rejects_path_with_null_byte(client: TestClient) -> None:
    response = client.post("/index", json={"paths": ["sample\x00.py"]})
    assert response.status_code == 400
