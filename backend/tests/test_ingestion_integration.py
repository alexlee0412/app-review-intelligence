"""PostgreSQL integration coverage for ingestion and embedding backfill."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.db import engine
from app.models.app import App
from app.models.review import EMBEDDING_DIMENSION, Review
from app.repositories.review_write_repository import upsert_normalized_records
from app.services.apify_source import load_from_file
from app.services.embedding_service import FakeEmbeddingProvider
from app.services.normalization import normalize_records
from scripts.generate_embeddings import backfill_embeddings

pytestmark = pytest.mark.integration

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "apify_sample.json"
SOURCE = "test:fixture"


@pytest.fixture
def session() -> Generator[Session, None, None]:
    try:
        connection = engine.connect()
    except SQLAlchemyError:
        pytest.skip("PostgreSQL is not available")

    transaction = connection.begin()
    db = Session(bind=connection)
    try:
        db.execute(delete(Review).where(Review.app_id.like("fixture-app-%")))
        db.execute(delete(App).where(App.app_id.like("fixture-app-%")))
        db.flush()
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()


def _normalized_fixture():
    return normalize_records(load_from_file(FIXTURE_PATH), source=SOURCE)


def test_fixture_ingest_is_idempotent_and_invalidates_changed_text(
    session: Session,
) -> None:
    result = _normalized_fixture()
    assert result.read_count == 12
    assert result.normalized_count == 7
    assert result.rejected_count == 5
    assert result.rejected_by_reason == {
        "EMPTY_BODY": 1,
        "INVALID_COUNTRY": 1,
        "INVALID_DATE": 1,
        "INVALID_RATING": 1,
        "MISSING_APP_ID": 1,
    }

    first = upsert_normalized_records(session, result.apps, result.reviews)
    session.flush()
    assert first.apps_inserted == 2
    assert first.reviews_inserted == 6
    assert session.scalar(
        select(func.count()).select_from(Review).where(Review.source == SOURCE)
    ) == 6

    preserved_id = f"{SOURCE}:fixture-r1"
    preserved_vector = FakeEmbeddingProvider().embed_query("preserve this vector")
    session.execute(
        update(Review)
        .where(Review.review_id == preserved_id)
        .values(embedding=preserved_vector)
    )
    session.flush()

    second = upsert_normalized_records(session, result.apps, result.reviews)
    session.flush()
    session.expire_all()
    assert second.apps_updated == 2
    assert second.reviews_updated == 6
    assert session.get(Review, preserved_id).embedding == pytest.approx(
        preserved_vector
    )

    changed_records = list(load_from_file(FIXTURE_PATH))
    changed_records[0] = {
        **changed_records[0],
        "text": "The body changed and the old vector is stale.",
    }
    changed = normalize_records(changed_records, source=SOURCE)
    upsert_normalized_records(session, changed.apps, changed.reviews)
    session.flush()
    session.expire_all()
    assert session.get(Review, preserved_id).embedding is None


def test_embedding_backfill_fills_only_null_rows(session: Session) -> None:
    result = _normalized_fixture()
    upsert_normalized_records(session, result.apps, result.reviews)
    session.flush()

    # Counted before the run: skipped is a whole-table figure, so the expectation has
    # to account for any rows the database already holds.
    preexisting_embedded = session.scalar(
        select(func.count()).select_from(Review).where(Review.embedding.is_not(None))
    )

    untouched_id = f"{SOURCE}:fixture-r2"
    untouched_vector = FakeEmbeddingProvider().embed_query("already embedded")
    session.execute(
        update(Review)
        .where(Review.review_id == untouched_id)
        .values(embedding=untouched_vector)
    )
    session.flush()

    connection = session.connection()
    batch_sessions = sessionmaker(
        bind=connection,
        join_transaction_mode="create_savepoint",
    )
    counts = backfill_embeddings(
        batch_sessions,
        FakeEmbeddingProvider(),
        batch_size=2,
    )

    session.expire_all()
    rows = session.scalars(
        select(Review).where(Review.source == SOURCE).order_by(Review.review_id)
    ).all()
    assert counts.embedded == 5
    assert counts.remaining == 0
    # The one fixture row seeded above, plus whatever was already embedded.
    assert counts.skipped == preexisting_embedded + 1
    assert len(rows) == 6
    assert all(row.embedding is not None for row in rows)
    assert all(len(row.embedding) == EMBEDDING_DIMENSION for row in rows)
    assert session.get(Review, untouched_id).embedding == pytest.approx(
        untouched_vector
    )
