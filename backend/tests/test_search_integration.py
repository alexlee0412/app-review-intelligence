"""Integration coverage for exact cosine review retrieval."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.db import engine
from app.models import EMBEDDING_DIMENSION, App, Review
from app.repositories.review_search_repository import search_review_candidates
from app.schemas.search import AppliedFilters

pytestmark = pytest.mark.integration

TEST_APP_IDS = ["search-app-a", "search-app-b"]


def _unit_vector(*components: tuple[int, float]) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    for index, value in components:
        vector[index] = value
    return vector


@pytest.fixture
def session() -> Generator[Session, None, None]:
    """Yield a database session whose transaction is always rolled back."""
    try:
        connection = engine.connect()
    except SQLAlchemyError:
        pytest.skip("PostgreSQL is not available")

    transaction = connection.begin()
    db = Session(bind=connection)
    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def seeded_session(session: Session) -> Session:
    session.add_all(
        [
            App(app_id="search-app-a", app_name="Search App A"),
            App(app_id="search-app-b", app_name="Search App B"),
        ]
    )
    session.flush()
    session.add_all(
        [
            Review(
                review_id="nearest-us-one",
                app_id="search-app-a",
                rating=1,
                country="US",
                created_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
                title="Nearest",
                body="Cannot cancel my subscription",
                embedding=_unit_vector((0, 1.0)),
            ),
            Review(
                review_id="second-kr-two",
                app_id="search-app-b",
                rating=2,
                country="KR",
                created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
                title=None,
                body="Subscription settings are confusing",
                embedding=_unit_vector((0, 0.8), (1, 0.6)),
            ),
            Review(
                review_id="boundary-us-two",
                app_id="search-app-a",
                rating=2,
                country="US",
                created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                title=None,
                body="Charged after cancellation",
                embedding=_unit_vector((1, 1.0)),
            ),
            Review(
                review_id="null-us-one",
                app_id="search-app-a",
                rating=1,
                country="US",
                created_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
                title=None,
                body="This review has no embedding",
                embedding=None,
            ),
        ]
    )
    session.flush()
    return session


def _search(session: Session, filters: AppliedFilters) -> object:
    scoped_filters = filters.model_copy(
        update={"app_ids": filters.app_ids or TEST_APP_IDS}
    )
    return search_review_candidates(
        session=session,
        query_vector=_unit_vector((0, 1.0)),
        filters=scoped_filters,
        top_k=20,
        candidate_multiplier=2,
    )


def test_knn_returns_nearest_first_and_excludes_null_embeddings(
    seeded_session: Session,
) -> None:
    result = _search(seeded_session, AppliedFilters())
    assert result.evidence[0].review_id == "nearest-us-one"
    assert "null-us-one" not in {item.review_id for item in result.evidence}


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        (AppliedFilters(countries=["KR"]), {"second-kr-two"}),
        (
            AppliedFilters(ratings=[2]),
            {"second-kr-two", "boundary-us-two"},
        ),
        (
            AppliedFilters(
                date_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
                date_to=datetime(2026, 9, 1, tzinfo=timezone.utc),
            ),
            {"nearest-us-one", "second-kr-two"},
        ),
        (
            AppliedFilters(
                app_ids=["search-app-a"],
                countries=["US"],
                ratings=[1, 2],
                date_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
                date_to=datetime(2026, 9, 1, tzinfo=timezone.utc),
            ),
            {"nearest-us-one"},
        ),
    ],
)
def test_filters_work_individually_and_in_combination(
    seeded_session: Session,
    filters: AppliedFilters,
    expected_ids: set[str],
) -> None:
    result = _search(seeded_session, filters)
    assert {item.review_id for item in result.evidence} == expected_ids


def test_matched_count_agrees_with_independent_count(seeded_session: Session) -> None:
    filters = AppliedFilters(
        countries=["US"],
        date_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
        date_to=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    result = _search(seeded_session, filters)
    expected = seeded_session.scalar(
        select(func.count())
        .select_from(Review)
        .where(
            Review.embedding.is_not(None),
            Review.app_id.in_(TEST_APP_IDS),
            Review.country.in_(filters.countries),
            Review.created_at >= filters.date_from,
            Review.created_at < filters.date_to,
        )
    )
    assert result.matched_review_count == expected


def test_identical_calls_have_identical_ordering(seeded_session: Session) -> None:
    first = _search(seeded_session, AppliedFilters())
    second = _search(seeded_session, AppliedFilters())
    assert [item.review_id for item in first.evidence] == [
        item.review_id for item in second.evidence
    ]


def test_min_similarity_is_applied_in_sql(seeded_session: Session) -> None:
    result = _search(seeded_session, AppliedFilters(min_similarity=0.9))
    assert [item.review_id for item in result.evidence] == ["nearest-us-one"]
    assert result.matched_review_count == 1


def test_sql_template_keeps_values_bound(seeded_session: Session) -> None:
    result = _search(
        seeded_session,
        AppliedFilters(
            app_ids=["search-app-a"],
            date_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
    )
    assert "search-app-a" not in result.sql_template
    assert "2026-06-01" not in result.sql_template
    assert "[1.0, 0.0" not in result.sql_template
    assert "query_embedding" in result.sql_template
