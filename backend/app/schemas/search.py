"""Schemas for filtered semantic review retrieval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

DEFAULT_TOP_K = 50
MAX_TOP_K = 200

CountryCode = Annotated[str, StringConstraints(min_length=2, max_length=2)]
Rating = Annotated[int, Field(ge=1, le=5)]


@dataclass(frozen=True)
class QueryEmbedder:
    """Minimal query-embedding contract used by the retrieval surface."""

    name: str
    is_production_grade: bool
    embed: Callable[[str], list[float]]


class SearchRequest(BaseModel):
    """A semantic search with a half-open date range, ``[date_from, date_to)``."""

    query: str = Field(min_length=1, max_length=512)
    app_ids: list[str] | None = Field(default=None, min_length=1)
    countries: list[CountryCode] | None = Field(default=None, min_length=1)
    ratings: list[Rating] | None = Field(default=None, min_length=1)
    date_from: datetime | None = None
    date_to: datetime | None = None
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @field_validator("app_ids")
    @classmethod
    def normalize_app_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [app_id.strip() for app_id in value]
        if any(not app_id for app_id in normalized):
            raise ValueError("app_ids must not contain blank values")
        return list(dict.fromkeys(normalized))

    @field_validator("countries", mode="before")
    @classmethod
    def normalize_countries(cls, value: object) -> object:
        if value is None or not isinstance(value, list):
            return value
        return [
            country.upper() if isinstance(country, str) else country
            for country in value
        ]

    @field_validator("countries")
    @classmethod
    def deduplicate_countries(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else list(dict.fromkeys(value))

    @field_validator("ratings")
    @classmethod
    def deduplicate_ratings(cls, value: list[int] | None) -> list[int] | None:
        return None if value is None else list(dict.fromkeys(value))

    @field_validator("date_from", "date_to")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_date_range(self) -> SearchRequest:
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from >= self.date_to
        ):
            raise ValueError("date_from must be earlier than date_to")
        return self


class AppliedFilters(BaseModel):
    """Normalized filters applied by the search query."""

    app_ids: list[str] | None = None
    countries: list[str] | None = None
    ratings: list[int] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_similarity: float | None = None


class ReviewEvidence(BaseModel):
    """A review returned as evidence for a semantic search."""

    review_id: str
    app_id: str
    app_name: str
    version: str | None
    rating: int
    country: str
    created_at: datetime
    title: str | None
    body: str
    similarity: float = Field(ge=0.0, le=1.0)


class QueryTrace(BaseModel):
    """Auditable details about retrieval execution."""

    sql_template: str
    top_k: int
    candidate_multiplier: int
    embedding_provider: str
    embedding_dimension: int
    similarity_metric: str


class SearchResponse(BaseModel):
    """Evidence-backed semantic retrieval results."""

    query_run_id: UUID
    matched_review_count: int
    returned_count: int
    warnings: list[str]
    applied_filters: AppliedFilters
    evidence: list[ReviewEvidence]
    query_trace: QueryTrace
