from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api import ask as ask_api
from app.core.db import get_db
from app.main import app, create_app
from app.schemas.answer import AnswerResponse, AnswerTrace, EvidenceItem, Finding
from app.schemas.search import AppliedFilters, ReviewEvidence
from app.services.llm_provider import (
    LLMConfigurationError,
    LLMError,
    LLMResponseError,
)
from app.services.review_search import QueryEmbeddingError


class FakeSession:
    pass


def _answer_response() -> AnswerResponse:
    filters = AppliedFilters(app_ids=["app-one"], countries=["US"], ratings=[1])
    review = ReviewEvidence(
        review_id="review-one",
        app_id="app-one",
        app_name="App One",
        version="1.0",
        rating=1,
        country="US",
        created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        title="Cancellation issue",
        body="Cannot cancel my subscription.",
        similarity=0.91,
    )
    evidence = EvidenceItem(
        evidence_id="E1",
        review=review,
        excerpt="Cannot cancel my subscription.",
    )
    return AnswerResponse(
        query_run_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        question="Why can users not cancel?",
        answer="A review reports a cancellation issue [E1].",
        findings=[
            Finding(
                claim="A review reports a cancellation issue.",
                evidence_ids=["E1"],
                kind="observed",
            )
        ],
        metrics={"totals": {"total_matched": 1}},
        evidence=[evidence],
        limitations=["The evidence is limited."],
        warnings=["A development provider was used."],
        trace=AnswerTrace(
            intent="semantic_evidence",
            applied_filters=filters,
            semantic_query="cancel subscription",
            total_candidates=1,
            evidence_count=1,
            aggregates_computed=["matched_count"],
            planner_model="planner-test",
            synthesizer_model="synthesizer-test",
            llm_provider="provider-test",
            llm_is_production_grade=False,
            embedding_provider="embedding-test",
            embedding_is_production_grade=False,
            synthesis_skipped=False,
        ),
    )


def _override_db(session: FakeSession) -> None:
    app.dependency_overrides[get_db] = lambda: session


def test_ask_returns_answer_contract_unchanged_and_calls_service_once(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    expected = _answer_response()
    calls: list[tuple[object, str]] = []

    def answer_question(received_session: object, question: str) -> AnswerResponse:
        calls.append((received_session, question))
        return expected

    _override_db(session)
    monkeypatch.setattr(ask_api, "answer_question", answer_question)

    response = client.post(
        "/api/v1/reviews/ask",
        json={"question": "Why can users not cancel?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert AnswerResponse.model_validate(body)
    assert body == json.loads(expected.model_dump_json())
    assert body["findings"] == json.loads(expected.model_dump_json())["findings"]
    assert body["metrics"] == expected.metrics
    assert body["evidence"] == json.loads(expected.model_dump_json())["evidence"]
    assert body["limitations"] == expected.limitations
    assert body["warnings"] == expected.warnings
    assert body["trace"] == json.loads(expected.model_dump_json())["trace"]
    assert calls == [(session, "Why can users not cancel?")]


@pytest.mark.parametrize("question", ["", " ", "\t\n"])
def test_ask_rejects_blank_question_without_calling_service(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    question: str,
) -> None:
    calls = 0

    def answer_question(*args: object, **kwargs: object) -> AnswerResponse:
        nonlocal calls
        calls += 1
        return _answer_response()

    _override_db(FakeSession())
    monkeypatch.setattr(ask_api, "answer_question", answer_question)

    response = client.post("/api/v1/reviews/ask", json={"question": question})

    assert response.status_code == 422
    assert calls == 0


_INTERNAL_DETAIL = (
    "secret-model secret-provider postgresql://user:password@db/app Traceback"
)


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (
            ValueError(_INTERNAL_DETAIL),
            400,
            "The question could not be interpreted.",
        ),
        (
            LLMConfigurationError(_INTERNAL_DETAIL),
            503,
            "Answer service is not configured.",
        ),
        (
            LLMResponseError(_INTERNAL_DETAIL),
            503,
            "Answer service is temporarily unavailable.",
        ),
        (
            LLMError(_INTERNAL_DETAIL),
            503,
            "Answer service is temporarily unavailable.",
        ),
        (
            QueryEmbeddingError(_INTERNAL_DETAIL),
            503,
            "Review search service is temporarily unavailable.",
        ),
        (
            SQLAlchemyError(_INTERNAL_DETAIL),
            503,
            "Review database is temporarily unavailable.",
        ),
    ],
)
def test_ask_maps_failures_without_exposing_internal_details(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    def answer_question(*args: object, **kwargs: object) -> AnswerResponse:
        raise error

    _override_db(FakeSession())
    monkeypatch.setattr(ask_api, "answer_question", answer_question)

    response = client.post(
        "/api/v1/reviews/ask",
        json={"question": "Why can users not cancel?"},
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    response_text = response.text.lower()
    for forbidden in (
        _INTERNAL_DETAIL.lower(),
        "secret-model",
        "secret-provider",
        "postgresql://",
        "password",
        "traceback",
    ):
        assert forbidden not in response_text


def test_cors_middleware_is_absent_without_configured_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(app_name="Test API", cors_allow_origins=[])
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    application = create_app()

    assert not any(
        middleware.cls is CORSMiddleware
        for middleware in application.user_middleware
    )


def test_cors_middleware_uses_configured_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = "https://frontend.example.invalid"
    settings = SimpleNamespace(app_name="Test API", cors_allow_origins=[origin])
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    application = create_app()

    assert any(
        middleware.cls is CORSMiddleware
        for middleware in application.user_middleware
    )
    response = TestClient(application).options(
        "/api/v1/reviews/ask",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
