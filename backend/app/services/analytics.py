"""Pure composition of deterministic per-app review aggregates."""

from __future__ import annotations

from math import fsum
from typing import Any

from app.schemas.answer import AppAggregate


def build_analytics_totals(aggregates: list[AppAggregate]) -> dict[str, Any]:
    """Combine per-app aggregates into bundle-level totals."""
    total_reviews = sum(item.review_count for item in aggregates)
    total_matched = sum(item.matched_count for item in aggregates)
    rating_distribution = {
        rating: sum(item.rating_distribution.get(rating, 0) for item in aggregates)
        for rating in range(1, 6)
    }
    rated_reviews = sum(
        item.review_count for item in aggregates if item.avg_rating is not None
    )
    weighted_rating = fsum(
        item.avg_rating * item.review_count
        for item in aggregates
        if item.avg_rating is not None
    )

    return {
        "total_reviews": total_reviews,
        "total_matched": total_matched,
        "apps_with_matches": sum(item.matched_count > 0 for item in aggregates),
        "overall_avg_rating": (
            weighted_rating / rated_reviews if rated_reviews else None
        ),
        "rating_distribution": rating_distribution,
    }
