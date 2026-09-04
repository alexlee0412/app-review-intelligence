from datetime import timezone

import pytest
from pydantic import ValidationError

from app.schemas.search import DEFAULT_TOP_K, MAX_TOP_K, SearchRequest


def test_countries_are_uppercased_and_deduplicated() -> None:
    request = SearchRequest(query="cancel", countries=["us", "US", "kr"])
    assert request.countries == ["US", "KR"]


def test_rating_range_is_enforced() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cancel", ratings=[0, 3, 6])


def test_ratings_are_deduplicated() -> None:
    assert SearchRequest(query="cancel", ratings=[1, 2, 1]).ratings == [1, 2]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(query: str) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query=query)


def test_query_length_is_bounded_after_trimming() -> None:
    assert len(SearchRequest(query="x" * 512).query) == 512
    with pytest.raises(ValidationError):
        SearchRequest(query="x" * 513)


@pytest.mark.parametrize("top_k", [0, MAX_TOP_K + 1])
def test_top_k_bounds_are_enforced(top_k: int) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cancel", top_k=top_k)


def test_top_k_has_documented_default() -> None:
    assert SearchRequest(query="cancel").top_k == DEFAULT_TOP_K


@pytest.mark.parametrize(
    ("date_from", "date_to"),
    [
        ("2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"),
        ("2026-06-02T00:00:00Z", "2026-06-01T00:00:00Z"),
    ],
)
def test_invalid_date_range_is_rejected(date_from: str, date_to: str) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cancel", date_from=date_from, date_to=date_to)


def test_naive_datetimes_are_assumed_utc() -> None:
    request = SearchRequest(
        query="cancel",
        date_from="2026-06-01T00:00:00",
        date_to="2026-07-01T00:00:00",
    )
    assert request.date_from is not None
    assert request.date_to is not None
    assert request.date_from.tzinfo is timezone.utc
    assert request.date_to.tzinfo is timezone.utc


@pytest.mark.parametrize("field", ["app_ids", "countries", "ratings"])
def test_present_filter_lists_must_not_be_empty(field: str) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cancel", **{field: []})
