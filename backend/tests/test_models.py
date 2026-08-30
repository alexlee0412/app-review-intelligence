"""Metadata-level tests for the ORM models. These never touch a database."""

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, Uuid, inspect
from sqlalchemy.dialects.postgresql import JSONB

from app.core.db import Base
from app.models import App, QueryRun, Review


def test_metadata_contains_core_tables() -> None:
    assert {"apps", "reviews", "query_runs"} <= set(Base.metadata.tables)


def test_review_rating_check_constraint() -> None:
    constraint = next(
        c
        for c in Review.__table__.constraints
        if isinstance(c, CheckConstraint) and c.name == "ck_reviews_rating_range"
    )
    assert "BETWEEN 1 AND 5" in str(constraint.sqltext)


def test_review_app_id_references_apps() -> None:
    foreign_key = next(iter(Review.__table__.c.app_id.foreign_keys))
    assert foreign_key.target_fullname == "apps.app_id"
    assert foreign_key.ondelete == "RESTRICT"
    assert Review.__table__.c.app_id.nullable is False


def test_review_embedding_dimension_is_1536() -> None:
    embedding = Review.__table__.c.embedding
    assert isinstance(embedding.type, Vector)
    assert embedding.type.dim == 1536


def test_review_embedding_is_nullable() -> None:
    assert Review.__table__.c.embedding.nullable is True


def test_query_run_primary_key_is_uuid() -> None:
    column = QueryRun.__table__.c.query_run_id
    assert isinstance(column.type, Uuid)
    assert column.primary_key is True
    assert isinstance(column.default.arg(None), uuid.UUID)


def test_json_columns_use_jsonb() -> None:
    for name in ("parsed_intent", "applied_filters", "result_summary"):
        assert isinstance(QueryRun.__table__.c[name].type, JSONB)


def test_timestamps_are_timezone_aware() -> None:
    columns = [
        App.__table__.c.created_at,
        Review.__table__.c.created_at,
        Review.__table__.c.ingested_at,
        QueryRun.__table__.c.created_at,
    ]
    for column in columns:
        assert isinstance(column.type, DateTime)
        assert column.type.timezone is True


def test_expected_indexes_exist() -> None:
    assert {index.name for index in App.__table__.indexes} == {"ix_apps_app_name"}
    assert {index.name for index in Review.__table__.indexes} == {
        "ix_reviews_country_rating_created_at",
        "ix_reviews_app_id",
    }


def test_app_review_relationship() -> None:
    app_relationship = inspect(App).relationships["reviews"]
    review_relationship = inspect(Review).relationships["app"]

    assert app_relationship.mapper.class_ is Review
    assert app_relationship.back_populates == "app"
    assert review_relationship.mapper.class_ is App
    assert review_relationship.back_populates == "reviews"


def test_review_created_at_has_no_server_default() -> None:
    """The review's own post date must always be supplied by the caller."""
    assert Review.__table__.c.created_at.server_default is None
    assert Review.__table__.c.ingested_at.server_default is not None
