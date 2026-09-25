"""Integration tests against a real PostgreSQL instance.

Skipped automatically when the database is unreachable, so the default test
run does not require Docker.
"""

from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.db import engine
from app.models import App, Review

pytestmark = pytest.mark.integration


@pytest.fixture
def session() -> Generator[Session, None, None]:
    """Yield a session whose work is always rolled back."""
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


def test_core_tables_exist(session: Session) -> None:
    table_names = set(inspect(session.get_bind()).get_table_names())
    assert {"apps", "reviews", "query_runs"} <= table_names


def test_query_run_instrumentation_columns_exist(session: Session) -> None:
    columns = {
        column["name"]
        for column in inspect(session.get_bind()).get_columns("query_runs")
    }
    assert {
        "run_kind",
        "stage_latency_ms",
        "model_usage",
        "grounding_outcomes",
        "retrieval_outcomes",
        "run_versions",
    } <= columns


def test_vector_extension_installed(session: Session) -> None:
    installed = session.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    ).scalar()
    assert installed == 1


def test_embedding_column_is_vector_1536(session: Session) -> None:
    udt_name, dimension = session.execute(
        text(
            """
            SELECT t.typname, a.atttypmod
            FROM pg_attribute a
            JOIN pg_type t ON t.oid = a.atttypid
            WHERE a.attrelid = 'reviews'::regclass AND a.attname = 'embedding'
            """
        )
    ).one()
    assert udt_name == "vector"
    assert dimension == 1536


def test_insert_review_with_null_embedding_and_relationship(session: Session) -> None:
    app = App(app_id="test-app", app_name="Test App", platform="ios")
    review = Review(
        review_id="test-review",
        app_id="test-app",
        rating=2,
        country="US",
        created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        body="Could not find how to cancel my subscription.",
    )
    session.add_all([app, review])
    session.flush()
    session.refresh(review)

    assert review.embedding is None
    assert review.app.app_name == "Test App"
    assert review.app.reviews == [review]
    # Server-side defaults populate on flush.
    assert review.ingested_at is not None
    assert app.created_at.tzinfo is not None


def test_invalid_rating_is_rejected(session: Session) -> None:
    session.add(App(app_id="bad-app", app_name="Bad App"))
    session.flush()

    with pytest.raises(IntegrityError), session.begin_nested():
        session.add(
            Review(
                review_id="bad-review",
                app_id="bad-app",
                rating=6,
                country="US",
                created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
                body="Rating is out of range.",
            )
        )
        session.flush()
