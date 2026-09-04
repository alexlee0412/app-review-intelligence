from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.deps import get_query_embedder
from app.core.db import get_db
from app.main import app
from app.models import EMBEDDING_DIMENSION
from app.schemas.search import QueryEmbedder
from app.services.embedding_service import EmbeddingConfigurationError


class EmptyResult:
    def all(self) -> list[object]:
        return []


class FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.commits = 0

    def execute(self, statement: object) -> EmptyResult:
        return EmptyResult()

    def add(self, value: object) -> None:
        self.added.append(value)

    def commit(self) -> None:
        self.commits += 1


def _override_dependencies(
    session: FakeSession,
    embedder: QueryEmbedder | None = None,
) -> None:
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_query_embedder] = lambda: embedder or QueryEmbedder(
        name="fake",
        is_production_grade=False,
        embed=lambda _: [0.0] * EMBEDDING_DIMENSION,
    )


def test_search_200_shape_matches_contract(client: TestClient) -> None:
    session = FakeSession()
    _override_dependencies(session)

    response = client.post(
        "/api/v1/reviews/search",
        json={"query": "cannot cancel", "countries": ["us"], "top_k": 10},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "query_run_id",
        "matched_review_count",
        "returned_count",
        "warnings",
        "applied_filters",
        "evidence",
        "query_trace",
    }
    assert set(body["query_trace"]) == {
        "sql_template",
        "top_k",
        "candidate_multiplier",
        "embedding_provider",
        "embedding_dimension",
        "similarity_metric",
    }
    assert body["applied_filters"]["countries"] == ["US"]
    assert body["query_trace"]["embedding_provider"] == "fake"
    assert body["warnings"]
    assert session.commits == 1


def test_search_bad_input_returns_422(client: TestClient) -> None:
    _override_dependencies(FakeSession())
    response = client.post("/api/v1/reviews/search", json={"query": "   "})
    assert response.status_code == 422


def test_search_embedding_failure_returns_sanitized_503(client: TestClient) -> None:
    secret_parts = ["postgres", "://user:", "password", "@db/", "app_review"]

    def fail(_: str) -> list[float]:
        raise RuntimeError("".join(secret_parts))

    _override_dependencies(
        FakeSession(),
        QueryEmbedder(name="broken", is_production_grade=False, embed=fail),
    )
    response = client.post("/api/v1/reviews/search", json={"query": "cancel"})

    assert response.status_code == 503
    body_text = response.text.lower()
    for forbidden in [
        "key",
        "token",
        "dsn",
        "postgres",
        "password",
        "@",
        "app_" + "review",
    ]:
        assert forbidden not in body_text


def test_search_empty_result_has_real_query_run_id(client: TestClient) -> None:
    session = FakeSession()
    _override_dependencies(session)
    response = client.post("/api/v1/reviews/search", json={"query": "no matches"})

    assert response.status_code == 200
    body = response.json()
    assert body["evidence"] == []
    assert body["matched_review_count"] == 0
    assert body["returned_count"] == 0
    assert body["query_run_id"]
    assert session.added[0].query_run_id is not None


def test_query_embedder_configuration_failure_returns_logged_503(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_build(_: object) -> object:
        raise EmbeddingConfigurationError("invalid configuration")

    monkeypatch.setattr("app.core.config.get_settings", lambda: object())
    monkeypatch.setattr(
        "app.services.embedding_service.build_embedding_provider", fail_build
    )

    with caplog.at_level(logging.ERROR, logger="app.api.deps"):
        with pytest.raises(HTTPException) as raised:
            get_query_embedder()

    assert raised.value.status_code == 503
    assert raised.value.detail == "Query embedding service unavailable"
    assert [record.message for record in caplog.records] == [
        "Embedding provider unavailable"
    ]


@pytest.mark.parametrize("error_type", [AttributeError, ImportError])
def test_query_embedder_programming_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    def fail_build(_: object) -> object:
        raise error_type("unexpected implementation failure")

    monkeypatch.setattr("app.core.config.get_settings", lambda: object())
    monkeypatch.setattr(
        "app.services.embedding_service.build_embedding_provider", fail_build
    )

    with pytest.raises(error_type, match="unexpected implementation failure"):
        get_query_embedder()
