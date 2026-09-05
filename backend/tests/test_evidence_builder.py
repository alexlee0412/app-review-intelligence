from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.schemas.answer import AppAggregate, MIN_EVIDENCE_FOR_SYNTHESIS
from app.schemas.query_plan import Intent, QueryPlan
from app.schemas.search import (
    AppliedFilters,
    QueryEmbedder,
    QueryTrace,
    ReviewEvidence,
    SearchResponse,
)
from app.services import evidence_builder
from app.services.evidence_builder import MAX_EXCERPT_CHARACTERS, build_evidence


def _review(index: int, *, body: str | None = None) -> ReviewEvidence:
    return ReviewEvidence(
        review_id=f"review-{index}",
        app_id="app-one",
        app_name="App One",
        version="1.0",
        rating=1,
        country="US",
        created_at=datetime(2026, 7, index, tzinfo=timezone.utc),
        title=None,
        body=body or f"Review body {index}",
        similarity=1.0 - index / 100,
    )


def _response(reviews: list[ReviewEvidence], *, production: bool = True) -> SearchResponse:
    return SearchResponse(
        query_run_id=uuid.uuid4(),
        matched_review_count=len(reviews),
        returned_count=len(reviews),
        warnings=(
            []
            if production
            else ["Development embeddings do not provide semantic ordering."]
        ),
        applied_filters=AppliedFilters(app_ids=["app-one"], countries=["US"]),
        evidence=reviews,
        query_trace=QueryTrace(
            sql_template="SELECT parameterized",
            top_k=8,
            candidate_multiplier=2,
            embedding_provider="openai" if production else "fake",
            embedding_dimension=1536,
            similarity_metric="cosine",
        ),
    )


def _plan(count: int = 8) -> QueryPlan:
    return QueryPlan(
        intent=Intent.SEMANTIC_EVIDENCE,
        semantic_query="subscription cancellation",
        app_ids=["app-one"],
        top_k=count,
    )


def _embedder(*, production: bool = True) -> QueryEmbedder:
    return QueryEmbedder(
        name="openai" if production else "fake",
        is_production_grade=production,
        embed=lambda _: [0.0] * 1536,
    )


def test_evidence_ids_are_deterministic_and_one_based(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviews = [_review(1), _review(2), _review(3)]
    monkeypatch.setattr(evidence_builder, "search_reviews", lambda *_: _response(reviews))

    first = build_evidence(object(), _plan(), embedder=_embedder())  # type: ignore[arg-type]
    second = build_evidence(object(), _plan(), embedder=_embedder())  # type: ignore[arg-type]

    assert [item.evidence_id for item in first.evidence] == ["E1", "E2", "E3"]
    assert first.evidence == second.evidence
    assert first.retrieval_trace.sql_template == "SELECT parameterized"


def test_excerpt_is_exact_source_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    body = "  Exact punctuation!\n" + "한" * MAX_EXCERPT_CHARACTERS
    review = _review(1, body=body)
    monkeypatch.setattr(
        evidence_builder, "search_reviews", lambda *_: _response([review])
    )

    bundle = build_evidence(object(), _plan(), embedder=_embedder())  # type: ignore[arg-type]

    assert bundle.evidence[0].excerpt == body[:MAX_EXCERPT_CHARACTERS]


def test_thin_and_nonproduction_evidence_limitations_are_carried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviews = [_review(index) for index in range(1, MIN_EVIDENCE_FOR_SYNTHESIS)]
    monkeypatch.setattr(
        evidence_builder,
        "search_reviews",
        lambda *_: _response(reviews, production=False),
    )

    bundle = build_evidence(
        object(), _plan(), embedder=_embedder(production=False)  # type: ignore[arg-type]
    )

    assert not bundle.has_sufficient_evidence
    assert any("Development embeddings" in item for item in bundle.limitations)
    assert any("at least 3" in item for item in bundle.limitations)
    assert bundle.retrieval_trace.embedding_is_production_grade is False


def test_aggregates_are_passed_through_without_recalculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviews = [_review(1), _review(2), _review(3)]
    aggregate = AppAggregate(
        app_id="app-one",
        app_name="App One",
        review_count=47,
        matched_count=3,
        avg_rating=1.75,
        rating_distribution={1: 2, 2: 1},
    )
    monkeypatch.setattr(evidence_builder, "search_reviews", lambda *_: _response(reviews))

    bundle = build_evidence(
        object(),  # type: ignore[arg-type]
        _plan(),
        embedder=_embedder(),
        aggregates=[aggregate],
    )

    assert bundle.aggregates == [aggregate]


def test_nonanswering_plan_does_not_call_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_: object) -> object:
        raise AssertionError("retrieval must be skipped")

    monkeypatch.setattr(evidence_builder, "search_reviews", fail)
    plan = QueryPlan(intent=Intent.UNSUPPORTED, semantic_query=None)

    bundle = build_evidence(object(), plan, embedder=_embedder())  # type: ignore[arg-type]

    assert bundle.evidence == []
    assert bundle.total_candidates == 0
    assert bundle.retrieval_trace.sql_template is None
