"""ORM model for the reviews table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.app import App

EMBEDDING_DIMENSION = 1536


class Review(Base):
    """A single app store review and its optional embedding."""

    __tablename__ = "reviews"

    __table_args__ = (
        CheckConstraint(
            "rating BETWEEN 1 AND 5", name="ck_reviews_rating_range"
        ),
        # Primary MVP filter: country + rating + date range.
        Index(
            "ix_reviews_country_rating_created_at",
            "country",
            "rating",
            "created_at",
        ),
        Index("ix_reviews_app_id", "app_id"),
        # No index on version: per-version analysis is a future extension, and
        # no approximate vector index on embedding: retrieval starts as exact
        # cosine search and will be measured before being optimized.
    )

    review_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    app_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("apps.app_id", name="fk_reviews_app_id_apps", ondelete="RESTRICT"),
        nullable=False,
    )
    version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    # ISO 3166-1 alpha-2 store country code.
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    # When the user originally posted the review.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # Nullable: reviews are ingested before embeddings are generated.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION), nullable=True
    )
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # When the review entered App Review Intelligence.
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    app: Mapped[App] = relationship(back_populates="reviews")
