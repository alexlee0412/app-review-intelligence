"""Structured plan produced from a natural-language question.

A plan is the only thing the question-interpretation step is allowed to emit. It is
validated and scope-clamped here, so downstream execution never acts on unchecked
model output: the country is forced to the supported storefront, unknown apps are
dropped rather than guessed at, and ``top_k`` is bounded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.search import MAX_TOP_K, CountryCode, Rating

# The dataset is US iOS App Store reviews only. A plan may narrow this scope but
# never widen it.
SUPPORTED_COUNTRIES: tuple[str, ...] = ("US",)
SUPPORTED_PLATFORM = "ios"

DEFAULT_PLAN_TOP_K = 8

RequestedMetric = Literal[
    "review_count",
    "matched_count",
    "avg_rating",
    "rating_distribution",
    "date_range",
]


class Intent(StrEnum):
    """The question shapes this milestone supports."""

    SEMANTIC_EVIDENCE = "semantic_evidence"
    APP_COMPARISON = "app_comparison"
    REVIEW_SUMMARY = "review_summary"
    FILTERED_EVIDENCE = "filtered_evidence"
    # Accepted so the planner can express it honestly, but the current dataset is far
    # too small for trend claims: execution reports insufficient evidence instead.
    TREND_ANALYSIS = "trend_analysis"
    UNSUPPORTED = "unsupported"


#: Intents that never reach synthesis with a substantive answer.
NON_ANSWERING_INTENTS: frozenset[Intent] = frozenset(
    {Intent.UNSUPPORTED, Intent.TREND_ANALYSIS}
)


class AppCatalogEntry(BaseModel):
    """One app the question may legitimately refer to.

    The catalog is supplied to the planner so it resolves names against reality
    rather than inventing identifiers.
    """

    app_id: str
    app_name: str


class QueryPlan(BaseModel):
    """A validated, scope-clamped interpretation of the user's question."""

    intent: Intent
    # None only when a plan needs aggregates alone.
    semantic_query: str | None = Field(default=None, max_length=512)
    app_ids: list[str] | None = Field(default=None, min_length=1)
    countries: list[CountryCode] = Field(default_factory=lambda: ["US"])
    ratings: list[Rating] | None = Field(default=None, min_length=1)
    # Half-open range, [date_from, date_to), matching retrieval everywhere else.
    date_from: datetime | None = None
    date_to: datetime | None = None
    top_k: int = Field(default=DEFAULT_PLAN_TOP_K, ge=1, le=MAX_TOP_K)
    group_by: Literal["app"] | None = None
    requested_metrics: list[RequestedMetric] = Field(default_factory=list)
    needs_semantic_search: bool = True
    needs_aggregation: bool = False
    # Planner commentary. Never rendered as fact.
    planner_notes: str | None = Field(default=None, max_length=500)

    @field_validator("semantic_query")
    @classmethod
    def normalize_semantic_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("app_ids")
    @classmethod
    def normalize_app_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [app_id.strip() for app_id in value if app_id and app_id.strip()]
        return list(dict.fromkeys(normalized)) or None

    @field_validator("countries", mode="before")
    @classmethod
    def uppercase_countries(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [c.upper() if isinstance(c, str) else c for c in value]

    @field_validator("countries")
    @classmethod
    def clamp_countries(cls, value: list[str]) -> list[str]:
        """Discard anything outside the supported storefront.

        A plan may not widen scope, so an unsupported country is dropped rather than
        honored, and an empty result falls back to the supported default.
        """
        supported = [c for c in dict.fromkeys(value) if c in SUPPORTED_COUNTRIES]
        return supported or list(SUPPORTED_COUNTRIES)

    @field_validator("ratings")
    @classmethod
    def deduplicate_ratings(cls, value: list[int] | None) -> list[int] | None:
        return None if value is None else list(dict.fromkeys(value))

    @field_validator("requested_metrics")
    @classmethod
    def deduplicate_metrics(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @field_validator("date_from", "date_to")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def enforce_execution_rules(self) -> QueryPlan:
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from >= self.date_to
        ):
            raise ValueError("date_from must be earlier than date_to")

        # Intents that cannot be answered from this dataset must not trigger retrieval.
        if self.intent in NON_ANSWERING_INTENTS:
            object.__setattr__(self, "needs_semantic_search", False)
            object.__setattr__(self, "needs_aggregation", False)

        # A semantic search without a query would silently return arbitrary rows.
        if self.needs_semantic_search and not self.semantic_query:
            raise ValueError("semantic_query is required when needs_semantic_search")

        if self.intent is Intent.APP_COMPARISON:
            object.__setattr__(self, "needs_aggregation", True)
            object.__setattr__(self, "group_by", "app")

        return self

    def restrict_apps_to(self, known_app_ids: set[str]) -> tuple[QueryPlan, list[str]]:
        """Drop app ids absent from the catalog, reporting what was discarded.

        Returns the restricted plan and a limitation message per dropped id, so an
        invented identifier becomes a visible caveat rather than an empty result.
        """
        if self.app_ids is None:
            return self, []
        kept = [app_id for app_id in self.app_ids if app_id in known_app_ids]
        dropped = [app_id for app_id in self.app_ids if app_id not in known_app_ids]
        limitations = [
            f"Ignored unknown app identifier {app_id!r}; it is not in the dataset."
            for app_id in dropped
        ]
        if kept == self.app_ids:
            return self, limitations
        return self.model_copy(update={"app_ids": kept or None}), limitations


class PlannerResult(BaseModel):
    """A plan plus how it was reached."""

    plan: QueryPlan
    model: str
    used_fallback: bool = False
    limitations: list[str] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
