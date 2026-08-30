"""ORM model for the apps table."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.review import Review


class App(Base):
    """A mobile app whose reviews are analyzed."""

    __tablename__ = "apps"

    app_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # Not unique: the same product may exist across platforms or stores.
    app_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    platform: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="ios"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # passive_deletes="all" defers to the ON DELETE RESTRICT foreign key, so
    # deleting an app that still has reviews raises a clear constraint error
    # instead of SQLAlchemy attempting to null out a non-nullable column.
    reviews: Mapped[list[Review]] = relationship(
        back_populates="app", passive_deletes="all"
    )
