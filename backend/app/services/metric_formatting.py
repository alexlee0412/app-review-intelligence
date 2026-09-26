"""Deterministic display formatting for computed answer metrics."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from app.schemas.answer import AppAggregate

UNAVAILABLE = "Not available"

_RATING_LABELS = {
    1: "one_star",
    2: "two_star",
    3: "three_star",
    4: "four_star",
    5: "five_star",
}


def format_count(
    value: int | None,
    *,
    singular: str | None = None,
    plural: str | None = None,
) -> str:
    """Format an integer count, optionally with a correctly inflected noun."""
    if value is None:
        return UNAVAILABLE
    display = f"{value:,}"
    if singular is None:
        return display
    noun = singular if value == 1 else (plural or f"{singular}s")
    return f"{display} {noun}"


def format_average(value: float | None) -> str:
    """Format an average to exactly two decimal places."""
    if value is None:
        return UNAVAILABLE
    return f"{value:.2f}"


def format_percentage(value: float | None) -> str:
    """Format a percentage-unit value to one decimal place."""
    if value is None:
        return UNAVAILABLE
    return f"{value:.1f}%"


def format_delta(value: int | None) -> str:
    """Format an integer delta with an explicit sign for positive values."""
    if value is None:
        return UNAVAILABLE
    return f"{value:+,}"


def format_date(value: date | datetime | None) -> str:
    """Format a date as a stable ISO calendar date."""
    if value is None:
        return UNAVAILABLE
    if isinstance(value, datetime):
        value = value.date()
    return value.isoformat()


def format_date_range(
    start: date | datetime | None,
    end: date | datetime | None,
) -> str:
    """Format a display range when both endpoints are available."""
    if start is None or end is None:
        return UNAVAILABLE
    return f"{format_date(start)} to {format_date(end)}"


def _format_distribution(distribution: Mapping[int, int]) -> dict[str, str]:
    return {
        label: format_count(
            distribution[rating], singular=f"{label.replace('_', '-')} review"
        )
        for rating, label in _RATING_LABELS.items()
        if rating in distribution
    }


def format_bundle_metrics(
    totals: Mapping[str, Any], aggregates: list[AppAggregate]
) -> dict[str, Any]:
    """Build display-only values parallel to the raw bundle metrics."""
    formatted_totals: dict[str, Any] = {}
    if "total_reviews" in totals:
        formatted_totals["total_reviews"] = format_count(
            totals["total_reviews"], singular="review"
        )
    if "total_matched" in totals:
        formatted_totals["total_matched"] = format_count(
            totals["total_matched"],
            singular="matching review",
            plural="matching reviews",
        )
    if "apps_with_matches" in totals:
        formatted_totals["apps_with_matches"] = format_count(
            totals["apps_with_matches"], singular="app"
        )
    if "overall_avg_rating" in totals:
        formatted_totals["overall_avg_rating"] = format_average(
            totals["overall_avg_rating"]
        )
    if "rating_distribution" in totals:
        formatted_totals["rating_distribution"] = _format_distribution(
            totals["rating_distribution"]
        )

    formatted_aggregates = [
        {
            "review_count": format_count(
                aggregate.review_count, singular="review"
            ),
            "matched_count": format_count(
                aggregate.matched_count,
                singular="matching review",
                plural="matching reviews",
            ),
            "avg_rating": format_average(aggregate.avg_rating),
            "rating_distribution": _format_distribution(
                aggregate.rating_distribution
            ),
            "oldest_review_at": format_date(aggregate.oldest_review_at),
            "newest_review_at": format_date(aggregate.newest_review_at),
        }
        for aggregate in aggregates
    ]
    return {"totals": formatted_totals, "aggregates": formatted_aggregates}
