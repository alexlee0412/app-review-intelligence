"""Deterministic SQL aggregates over filtered review rows."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import column, func, literal, select, table
from sqlalchemy.orm import Session

from app.schemas.answer import AppAggregate

_APPS = table(
    "apps",
    column("app_id"),
    column("app_name"),
)
_REVIEWS = table(
    "reviews",
    column("review_id"),
    column("app_id"),
    column("rating"),
    column("country"),
    column("created_at"),
)


def aggregate_reviews_by_app(
    session: Session,
    *,
    app_ids: list[str] | None = None,
    countries: list[str] | None = None,
    ratings: list[int] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    matched_review_ids: set[str] | None = None,
) -> list[AppAggregate]:
    """Return per-app aggregates for one bound, half-open filter selection."""
    if matched_review_ids:
        matched_count = func.count().filter(
            _REVIEWS.c.review_id.in_(sorted(matched_review_ids))
        )
    else:
        matched_count = literal(0)

    rating_counts = [
        func.count().filter(_REVIEWS.c.rating == rating).label(
            f"rating_{rating}"
        )
        for rating in range(1, 6)
    ]
    statement = (
        select(
            _REVIEWS.c.app_id,
            _APPS.c.app_name,
            func.count().label("review_count"),
            matched_count.label("matched_count"),
            func.avg(_REVIEWS.c.rating).label("avg_rating"),
            *rating_counts,
            func.min(_REVIEWS.c.created_at).label("oldest_review_at"),
            func.max(_REVIEWS.c.created_at).label("newest_review_at"),
        )
        .select_from(_REVIEWS.join(_APPS, _APPS.c.app_id == _REVIEWS.c.app_id))
        .group_by(_REVIEWS.c.app_id, _APPS.c.app_name)
        .order_by(_REVIEWS.c.app_id.asc(), _APPS.c.app_name.asc())
    )

    if app_ids is not None:
        statement = statement.where(_REVIEWS.c.app_id.in_(app_ids))
    if countries is not None:
        statement = statement.where(_REVIEWS.c.country.in_(countries))
    if ratings is not None:
        statement = statement.where(_REVIEWS.c.rating.in_(ratings))
    if date_from is not None:
        statement = statement.where(_REVIEWS.c.created_at >= date_from)
    if date_to is not None:
        statement = statement.where(_REVIEWS.c.created_at < date_to)

    aggregates: list[AppAggregate] = []
    for row in session.execute(statement):
        values = row._mapping
        aggregates.append(
            AppAggregate(
                app_id=str(values["app_id"]),
                app_name=str(values["app_name"]),
                review_count=int(values["review_count"]),
                matched_count=int(values["matched_count"]),
                avg_rating=(
                    float(values["avg_rating"])
                    if values["avg_rating"] is not None
                    else None
                ),
                rating_distribution={
                    rating: int(values[f"rating_{rating}"])
                    for rating in range(1, 6)
                },
                oldest_review_at=values["oldest_review_at"],
                newest_review_at=values["newest_review_at"],
            )
        )
    return aggregates
