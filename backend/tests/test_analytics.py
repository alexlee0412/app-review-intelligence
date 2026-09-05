"""Unit tests for pure aggregate composition."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.schemas.answer import AppAggregate
from app.services.analytics import build_analytics_totals


def test_build_analytics_totals_weights_ratings_and_combines_counts() -> None:
    aggregates = [
        AppAggregate(
            app_id="app-a",
            app_name="App A",
            review_count=2,
            matched_count=1,
            avg_rating=2.0,
            rating_distribution={1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            oldest_review_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            newest_review_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        AppAggregate(
            app_id="app-b",
            app_name="App B",
            review_count=3,
            matched_count=2,
            avg_rating=4.0,
            rating_distribution={1: 0, 2: 0, 3: 0, 4: 3, 5: 0},
        ),
    ]

    totals = build_analytics_totals(aggregates)

    assert totals["total_reviews"] == 5
    assert totals["total_matched"] == 3
    assert totals["apps_with_matches"] == 2
    assert totals["overall_avg_rating"] == pytest.approx(3.2)
    assert totals["rating_distribution"] == {1: 1, 2: 0, 3: 1, 4: 3, 5: 0}


def test_build_analytics_totals_handles_empty_selection() -> None:
    assert build_analytics_totals([]) == {
        "total_reviews": 0,
        "total_matched": 0,
        "apps_with_matches": 0,
        "overall_avg_rating": None,
        "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    }
