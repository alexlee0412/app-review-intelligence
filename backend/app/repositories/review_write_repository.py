"""Chunked PostgreSQL upserts for normalized apps and reviews."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from sqlalchemy import case, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.app import App
from app.models.review import Review
from app.schemas.ingestion import NormalizedApp, NormalizedReview

_Record = TypeVar("_Record", NormalizedApp, NormalizedReview)


@dataclass(frozen=True)
class WriteCounts:
    """Inserted and updated row counts for both tables."""

    apps_inserted: int = 0
    apps_updated: int = 0
    reviews_inserted: int = 0
    reviews_updated: int = 0

    @property
    def inserted(self) -> int:
        return self.apps_inserted + self.reviews_inserted

    @property
    def updated(self) -> int:
        return self.apps_updated + self.reviews_updated


def _chunks(records: Sequence[_Record], size: int) -> Iterable[Sequence[_Record]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _deduplicate(
    records: Iterable[_Record], key: str
) -> list[_Record]:
    by_id: dict[str, _Record] = {}
    for record in records:
        by_id[str(getattr(record, key))] = record
    return list(by_id.values())


def upsert_normalized_records(
    session: Session,
    apps: Iterable[NormalizedApp],
    reviews: Iterable[NormalizedReview],
    *,
    chunk_size: int = 500,
) -> WriteCounts:
    """Upsert applications before reviews and preserve valid embeddings."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    app_rows = _deduplicate(apps, "app_id")
    review_rows = _deduplicate(reviews, "review_id")
    apps_inserted = 0
    apps_updated = 0
    reviews_inserted = 0
    reviews_updated = 0

    for chunk in _chunks(app_rows, chunk_size):
        ids = [row.app_id for row in chunk]
        existing = set(
            session.scalars(select(App.app_id).where(App.app_id.in_(ids))).all()
        )
        apps_updated += len(existing)
        apps_inserted += len(ids) - len(existing)

        statement = insert(App).values(
            [row.model_dump() for row in chunk]
        )
        statement = statement.on_conflict_do_update(
            index_elements=[App.app_id],
            set_={
                "app_name": statement.excluded.app_name,
                "category": statement.excluded.category,
                "platform": statement.excluded.platform,
            },
        )
        session.execute(statement)

    for chunk in _chunks(review_rows, chunk_size):
        ids = [row.review_id for row in chunk]
        existing = set(
            session.scalars(select(Review.review_id).where(Review.review_id.in_(ids))).all()
        )
        reviews_updated += len(existing)
        reviews_inserted += len(ids) - len(existing)

        statement = insert(Review).values(
            [row.model_dump() for row in chunk]
        )
        text_changed = or_(
            Review.body.is_distinct_from(statement.excluded.body),
            Review.title.is_distinct_from(statement.excluded.title),
        )
        statement = statement.on_conflict_do_update(
            index_elements=[Review.review_id],
            set_={
                "version": statement.excluded.version,
                "rating": statement.excluded.rating,
                "country": statement.excluded.country,
                "created_at": statement.excluded.created_at,
                "title": statement.excluded.title,
                "body": statement.excluded.body,
                "source": statement.excluded.source,
                "embedding": case(
                    (text_changed, None),
                    else_=Review.embedding,
                ),
            },
        )
        session.execute(statement)

    return WriteCounts(
        apps_inserted=apps_inserted,
        apps_updated=apps_updated,
        reviews_inserted=reviews_inserted,
        reviews_updated=reviews_updated,
    )


class ReviewWriteRepository:
    """Small session-bound facade for the ingestion script."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self,
        apps: Iterable[NormalizedApp],
        reviews: Iterable[NormalizedReview],
        *,
        chunk_size: int = 500,
    ) -> WriteCounts:
        return upsert_normalized_records(
            self.session,
            apps,
            reviews,
            chunk_size=chunk_size,
        )
