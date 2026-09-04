"""Validated records and summaries produced by review normalization."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NormalizedApp(BaseModel):
    """An application row ready for persistence."""

    model_config = ConfigDict(extra="forbid")

    app_id: str
    app_name: str
    category: str | None = None
    platform: str = "ios"


class NormalizedReview(BaseModel):
    """A privacy-minimized review row ready for persistence."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    app_id: str
    version: str | None = None
    rating: int
    country: str
    created_at: datetime
    title: str | None = None
    body: str
    source: str


class RejectedRecord(BaseModel):
    """A source record that could not be normalized."""

    model_config = ConfigDict(extra="forbid")

    reason_code: str
    message: str
    raw: dict[str, Any]


class NormalizationResult(BaseModel):
    """Normalized rows, rejected rows, and aggregate processing counts."""

    model_config = ConfigDict(extra="forbid")

    apps: list[NormalizedApp] = Field(default_factory=list)
    reviews: list[NormalizedReview] = Field(default_factory=list)
    rejects: list[RejectedRecord] = Field(default_factory=list)
    read_count: int = 0
    normalized_count: int = 0
    rejected_count: int = 0
    rejected_by_reason: dict[str, int] = Field(default_factory=dict)
