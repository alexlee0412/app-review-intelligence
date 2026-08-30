"""ORM model for the query_runs table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class QueryRun(Base):
    """Traceability record for a single executed analysis.

    Stores how an analysis was parameterized and what it produced, so results
    remain auditable. Never store credentials, API keys, or connection strings
    in these columns.
    """

    __tablename__ = "query_runs"

    # Generated application-side so the SQL schema needs no default and no
    # extension for UUID generation.
    query_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Nullable: the first analysis flow uses fixed parameters rather than a
    # natural-language request.
    user_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_intent: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    applied_filters: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    sql_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
