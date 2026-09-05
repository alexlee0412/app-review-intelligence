"""PostgreSQL integration coverage for deterministic review aggregates."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.db import engine
from app.models import App, Review
from app.repositories.review_analytics_repository import aggregate_reviews_by_app

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class SeededAnalyticsRows:
    app_ids: list[str]
    review_ids: dict[str, str]


@pytest.fixture
def session() -> Generator[Session, None, None]:
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
def seeded_rows(session: Session) -> SeededAnalyticsRows:
    suffix = uuid4().hex[:12]
    app_a = f"analytics-a-{suffix}"
    app_b = f"analytics-b-{suffix}"
    review_ids = {
        name: f"{name}-{suffix}"
        for name in (
            "a-before",
            "a-start",
            "a-middle",
            "a-end",
            "a-kr",
            "b-middle",
            "b-late",
        )
    }
    session.add_all(
        [
            App(app_id=app_a, app_name="Analytics App A"),
            App(app_id=app_b, app_name="Analytics App B"),
        ]
    )
    session.flush()
    session.add_all(
        [
            Review(
                review_id=review_ids["a-before"],
                app_id=app_a,
                rating=5,
                country="US",
                created_at=datetime(2026, 5, 31, 23, 59, tzinfo=timezone.utc),
                body="Before range",
            ),
            Review(
                review_id=review_ids["a-start"],
                app_id=app_a,
                rating=1,
                country="US",
                created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                body="Inclusive start",
            ),
            Review(
                review_id=review_ids["a-middle"],
                app_id=app_a,
                rating=2,
                country="US",
                created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
                body="Middle",
            ),
            Review(
                review_id=review_ids["a-end"],
                app_id=app_a,
                rating=3,
                country="US",
                created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                body="Exclusive end",
            ),
            Review(
                review_id=review_ids["a-kr"],
                app_id=app_a,
                rating=4,
                country="KR",
                created_at=datetime(2026, 7, 2, tzinfo=timezone.utc),
                body="Other storefront",
            ),
            Review(
                review_id=review_ids["b-middle"],
                app_id=app_b,
                rating=4,
                country="US",
                created_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
                body="Middle B",
            ),
            Review(
                review_id=review_ids["b-late"],
                app_id=app_b,
                rating=5,
                country="US",
                created_at=datetime(2026, 8, 31, 23, 59, tzinfo=timezone.utc),
                body="Late B",
            ),
        ]
    )
    session.flush()
    return SeededAnalyticsRows(
        app_ids=[app_a, app_b],
        review_ids=review_ids,
    )


def _date_range() -> tuple[datetime, datetime]:
    return (
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def test_aggregate_count_matches_independent_sql(
    session: Session,
    seeded_rows: SeededAnalyticsRows,
) -> None:
    date_from, date_to = _date_range()
    matched_ids = {
        seeded_rows.review_ids["a-middle"],
        seeded_rows.review_ids["b-middle"],
    }
    aggregates = aggregate_reviews_by_app(
        session,
        app_ids=seeded_rows.app_ids,
        countries=["US"],
        date_from=date_from,
        date_to=date_to,
        matched_review_ids=matched_ids,
    )
    expected_count = session.scalar(
        select(func.count())
        .select_from(Review)
        .where(
            Review.app_id.in_(seeded_rows.app_ids),
            Review.country == "US",
            Review.created_at >= date_from,
            Review.created_at < date_to,
        )
    )

    assert sum(item.review_count for item in aggregates) == expected_count == 4
    assert sum(item.matched_count for item in aggregates) == 2
    by_app = {item.app_id: item for item in aggregates}
    app_a = by_app[seeded_rows.app_ids[0]]
    assert app_a.review_count == 2
    assert app_a.avg_rating == pytest.approx(1.5)
    assert app_a.rating_distribution == {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
    assert app_a.oldest_review_at == date_from
    assert app_a.newest_review_at == datetime(
        2026, 7, 1, tzinfo=timezone.utc
    )


def test_filters_and_half_open_boundaries_are_respected(
    session: Session,
    seeded_rows: SeededAnalyticsRows,
) -> None:
    date_from, date_to = _date_range()
    aggregates = aggregate_reviews_by_app(
        session,
        app_ids=seeded_rows.app_ids,
        countries=["US"],
        ratings=[2, 4],
        date_from=date_from,
        date_to=date_to,
    )

    assert [(item.app_id, item.review_count) for item in aggregates] == [
        (seeded_rows.app_ids[0], 1),
        (seeded_rows.app_ids[1], 1),
    ]
    assert aggregates[0].rating_distribution == {
        1: 0,
        2: 1,
        3: 0,
        4: 0,
        5: 0,
    }
    assert aggregates[1].rating_distribution == {
        1: 0,
        2: 0,
        3: 0,
        4: 1,
        5: 0,
    }


def test_ordering_is_deterministic(
    session: Session,
    seeded_rows: SeededAnalyticsRows,
) -> None:
    first = aggregate_reviews_by_app(
        session,
        app_ids=list(reversed(seeded_rows.app_ids)),
    )
    second = aggregate_reviews_by_app(
        session,
        app_ids=list(reversed(seeded_rows.app_ids)),
    )
    first_ids = [item.app_id for item in first]

    assert first_ids == sorted(seeded_rows.app_ids)
    assert first_ids == [item.app_id for item in second]
