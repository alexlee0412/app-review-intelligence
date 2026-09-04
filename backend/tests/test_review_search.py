from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models import EMBEDDING_DIMENSION, QueryRun
from app.repositories.review_search_repository import (
    MAX_CANDIDATE_LIMIT,
    SearchQueryResult,
    search_review_candidates,
)
from app.schemas.search import (
    MAX_TOP_K,
    AppliedFilters,
    QueryEmbedder,
    ReviewEvidence,
    SearchRequest,
)
from app.services import review_search
from app.services.review_search import InvalidEmbeddingDimensionError, search_reviews


class FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.commit_count = 0

    def add(self, value: object) -> None:
        self.added.append(value)

    def commit(self) -> None:
        self.commit_count += 1


class EmptyResult:
    def all(self) -> list[object]:
        return []


class CapturingSession:
    def __init__(self) -> None:
        self.statement: object | None = None

    def execute(self, statement: object) -> EmptyResult:
        self.statement = statement
        return EmptyResult()


def _embedder(*, production: bool = True) -> QueryEmbedder:
    return QueryEmbedder(
        name="openai" if production else "fake",
        is_production_grade=production,
        embed=lambda _: [0.0] * EMBEDDING_DIMENSION,
    )


def _evidence(
    review_id: str,
    body: str,
    similarity: float,
    created_at: datetime,
    app_id: str = "example.app",
) -> ReviewEvidence:
    return ReviewEvidence.model_construct(
        review_id=review_id,
        app_id=app_id,
        app_name="Example",
        version="1.0",
        rating=1,
        country="US",
        created_at=created_at,
        title=None,
        body=body,
        similarity=similarity,
    )


def _install_result(monkeypatch: pytest.MonkeyPatch, result: SearchQueryResult) -> None:
    monkeypatch.setattr(review_search, "search_review_candidates", lambda **_: result)


def test_similarity_is_clamped_rounded_and_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_result(
        monkeypatch,
        SearchQueryResult(
            evidence=[
                _evidence("low", "low", -0.2, now),
                _evidence("high", "high", 1.2, now),
                _evidence("rounded", "rounded", 0.12345678, now),
            ],
            matched_review_count=3,
            sql_template="SELECT parameterized",
        ),
    )

    response = search_reviews(FakeSession(), SearchRequest(query="cancel"), _embedder())

    assert [item.review_id for item in response.evidence] == [
        "high",
        "rounded",
        "low",
    ]
    assert [item.similarity for item in response.evidence] == [1.0, 0.123457, 0.0]


def test_dedup_keeps_best_member_and_count_remains_pre_dedup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older = datetime(2026, 6, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_result(
        monkeypatch,
        SearchQueryResult(
            evidence=[
                _evidence("lower", "Cannot cancel!!!", 0.7, newer),
                _evidence("best", " cannot   cancel ", 0.9, older),
                _evidence("other", "Billing failed", 0.8, newer),
            ],
            matched_review_count=17,
            sql_template="SELECT parameterized",
        ),
    )

    response = search_reviews(FakeSession(), SearchRequest(query="cancel"), _embedder())

    assert response.matched_review_count == 17
    assert response.returned_count == 2
    assert [item.review_id for item in response.evidence] == ["best", "other"]


def test_dedup_tiebreaks_by_recency_then_review_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older = datetime(2026, 6, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_result(
        monkeypatch,
        SearchQueryResult(
            evidence=[
                _evidence("old", "Same body", 0.8, older),
                _evidence("z-id", "same body!", 0.8, newer),
                _evidence("a-id", "same body.", 0.8, newer),
            ],
            matched_review_count=3,
            sql_template="SELECT parameterized",
        ),
    )

    response = search_reviews(FakeSession(), SearchRequest(query="cancel"), _embedder())
    assert [item.review_id for item in response.evidence] == ["a-id"]


def test_wrong_embedding_dimension_raises_typed_error() -> None:
    embedder = QueryEmbedder(
        name="broken",
        is_production_grade=False,
        embed=lambda _: [0.0],
    )
    with pytest.raises(InvalidEmbeddingDimensionError):
        search_reviews(FakeSession(), SearchRequest(query="cancel"), embedder)


def test_query_run_is_populated_without_embedding_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    _install_result(
        monkeypatch,
        SearchQueryResult(
            evidence=[], matched_review_count=0, sql_template="SELECT parameterized"
        ),
    )

    response = search_reviews(
        session,
        SearchRequest(query=" cancel ", countries=["us"], ratings=[1, 1]),
        _embedder(),
    )

    assert session.commit_count == 1
    assert len(session.added) == 1
    query_run = session.added[0]
    assert isinstance(query_run, QueryRun)
    assert query_run.query_run_id == response.query_run_id
    assert query_run.user_query == "cancel"
    assert query_run.parsed_intent is None
    assert query_run.applied_filters == {
        "app_ids": None,
        "countries": ["US"],
        "ratings": [1],
        "date_from": None,
        "date_to": None,
        "min_similarity": None,
    }
    assert query_run.sql_template == "SELECT parameterized"
    assert query_run.result_summary == {
        "matched_review_count": 0,
        "returned_count": 0,
        "top_similarity": None,
        "app_ids_returned": [],
        "embedding_provider": "openai",
    }
    assert str([query_run.applied_filters, query_run.result_summary]).count("0.0, 0.0") == 0


def test_development_provider_has_prominent_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_result(
        monkeypatch,
        SearchQueryResult([], 0, "SELECT parameterized"),
    )
    response = search_reviews(
        FakeSession(), SearchRequest(query="cancel"), _embedder(production=False)
    )
    assert response.warnings
    assert "non-semantic" in response.warnings[0]
    assert "NOT meaningful" in response.warnings[0]


def test_production_provider_has_no_quality_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_result(
        monkeypatch,
        SearchQueryResult([], 0, "SELECT parameterized"),
    )
    response = search_reviews(FakeSession(), SearchRequest(query="cancel"), _embedder())
    assert response.warnings == []


def test_candidate_limit_preserves_headroom_at_maximum_top_k() -> None:
    session = CapturingSession()

    search_review_candidates(
        session=session,  # type: ignore[arg-type]
        query_vector=[0.0] * EMBEDDING_DIMENSION,
        filters=AppliedFilters(),
        top_k=MAX_TOP_K,
        candidate_multiplier=2,
    )

    assert session.statement is not None
    assert session.statement._limit_clause.value == MAX_CANDIDATE_LIMIT  # type: ignore[attr-defined]
    assert MAX_CANDIDATE_LIMIT == MAX_TOP_K * 2


def test_maximum_top_k_is_satisfied_after_deduplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    candidates = [
        _evidence(
            review_id=f"review-{index}-{copy}",
            body=f"unique body {index}",
            similarity=1.0 - index / 1000,
            created_at=now,
        )
        for index in range(MAX_TOP_K)
        for copy in range(2)
    ]
    _install_result(
        monkeypatch,
        SearchQueryResult(
            evidence=candidates,
            matched_review_count=len(candidates),
            sql_template="SELECT parameterized",
        ),
    )

    response = search_reviews(
        FakeSession(),
        SearchRequest(query="cancel", top_k=MAX_TOP_K),
        _embedder(),
    )

    assert response.returned_count == MAX_TOP_K
    assert response.warnings == []


def test_deduplication_shortfall_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _install_result(
        monkeypatch,
        SearchQueryResult(
            evidence=[
                _evidence(
                    f"review-{index}",
                    f"same body{'!' * index}",
                    0.9 - index / 100,
                    now,
                )
                for index in range(10)
            ],
            matched_review_count=10,
            sql_template="SELECT parameterized",
        ),
    )

    response = search_reviews(
        FakeSession(), SearchRequest(query="cancel", top_k=5), _embedder()
    )

    assert response.returned_count == 1
    assert len(response.warnings) == 1
    assert "Returned 1 unique reviews for top_k=5" in response.warnings[0]


def test_provider_module_is_only_imported_inside_dependency_function() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    deps_source = (backend_root / "app/api/deps.py").read_text()
    service_source = (backend_root / "app/services/review_search.py").read_text()
    provider_module = "app.services." + "embedding_service"

    assert provider_module not in service_source
    import_line = "from " + provider_module + " import build_embedding_provider"
    assert import_line in deps_source
    assert deps_source.index(import_line) > deps_source.index("def get_query_embedder")


def test_owned_files_have_no_machine_specific_runtime_values() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    relative_paths = [
        "app/schemas/search.py",
        "app/repositories/review_search_repository.py",
        "app/services/review_search.py",
        "app/api/deps.py",
        "app/api/search.py",
        "app/main.py",
        "tests/test_search_schemas.py",
        "tests/test_review_search.py",
        "tests/test_search_api.py",
        "tests/test_search_integration.py",
    ]
    forbidden = [
        "local" + "host",
        "127" + ".0.0.1",
        "/" + "Users/",
        ":" + "8000",
    ]
    for relative_path in relative_paths:
        source = (backend_root / relative_path).read_text()
        assert all(value not in source for value in forbidden), relative_path
