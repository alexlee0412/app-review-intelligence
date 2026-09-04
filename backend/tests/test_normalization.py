"""Unit tests for tolerant, deterministic review normalization."""

from __future__ import annotations

from datetime import timezone

import pytest

from app.services.normalization import normalize_record, normalize_records


def valid_record() -> dict[str, object]:
    return {
        "id": "review-1",
        "appId": "app-1",
        "appName": "Example App",
        "score": 4,
        "country": "us",
        "date": "2026-02-03T10:15:00Z",
        "title": "  Helpful\r\n title  ",
        "text": "  First line\r\n  second line  ",
        "version": "1.2.3",
    }


@pytest.mark.parametrize("rating", [1, "2", 3.0, "4.0", 5])
def test_rating_is_coerced(rating: object) -> None:
    raw = valid_record()
    raw["score"] = rating
    _, review = normalize_record(raw, source="apify:test")
    assert review.rating == int(float(str(rating)))


@pytest.mark.parametrize("rating", [None, 0, 6, 2.5, "bad", True])
def test_invalid_rating_is_rejected(rating: object) -> None:
    raw = valid_record()
    raw["score"] = rating
    result = normalize_records([raw], source="apify:test")
    assert result.rejected_by_reason == {"INVALID_RATING": 1}


@pytest.mark.parametrize("country", ["U", "USA", "1A", "éS", ""])
def test_invalid_country_is_rejected(country: str) -> None:
    raw = valid_record()
    raw["country"] = country
    result = normalize_records([raw], source="apify:test")
    assert result.rejects[0].reason_code == "INVALID_COUNTRY"


def test_country_and_text_are_normalized_without_collapsing_internal_space() -> None:
    _, review = normalize_record(valid_record(), source="apify:test")
    assert review.country == "US"
    assert review.title == "Helpful\n title"
    assert review.body == "First line\n  second line"


@pytest.mark.parametrize(
    ("value", "expected_hour"),
    [
        ("2026-02-03T10:15:00Z", 10),
        ("2026-02-03T19:15:00+09:00", 10),
        ("2026-02-03T10:15:00", 10),
    ],
)
def test_iso_dates_become_utc(value: str, expected_hour: int) -> None:
    raw = valid_record()
    raw["date"] = value
    _, review = normalize_record(raw, source="apify:test")
    assert review.created_at.tzinfo == timezone.utc
    assert review.created_at.hour == expected_hour


def test_invalid_date_is_rejected() -> None:
    raw = valid_record()
    raw["date"] = "yesterday-ish"
    result = normalize_records([raw], source="apify:test")
    assert result.rejects[0].reason_code == "INVALID_DATE"


def test_empty_title_becomes_none_and_empty_body_is_rejected() -> None:
    raw = valid_record()
    raw["title"] = " \r\n "
    _, review = normalize_record(raw, source="apify:test")
    assert review.title is None

    raw["text"] = " \r\n "
    result = normalize_records([raw], source="apify:test")
    assert result.rejects[0].reason_code == "EMPTY_BODY"


def test_version_is_stripped_and_truncated() -> None:
    raw = valid_record()
    raw["version"] = "  " + "v" * 40 + "  "
    _, review = normalize_record(raw, source="apify:test")
    assert review.version == "v" * 32


def test_missing_app_id_rejects_unless_override_is_present() -> None:
    raw = valid_record()
    del raw["appId"]
    result = normalize_records([raw], source="apify:test")
    assert result.rejects[0].reason_code == "MISSING_APP_ID"

    app, review = normalize_record(
        raw, source="apify:test", app_id_override="override-app"
    )
    assert app.app_id == review.app_id == "override-app"


def test_review_id_fallback_is_deterministic() -> None:
    raw = valid_record()
    del raw["id"]
    _, first = normalize_record(raw, source="apify:test")
    _, second = normalize_record(dict(raw), source="apify:test")
    assert first.review_id == second.review_id
    assert len(first.review_id) <= 64


def test_long_external_id_uses_deterministic_fallback() -> None:
    raw = valid_record()
    raw["id"] = "x" * 100
    _, review = normalize_record(raw, source="s" * 80)
    assert review.review_id.startswith("s" * 24 + ":")
    assert len(review.review_id) == 57
    assert review.source == "s" * 64


def test_pii_is_absent_from_normalized_objects() -> None:
    raw = valid_record()
    raw.update(
        {
            "userName": "private",
            "userUrl": "https://invalid.example/private",
            "reviewerId": "private-id",
            "avatarUrl": "https://invalid.example/avatar",
        }
    )
    app, review = normalize_record(raw, source="apify:test")
    combined = {**app.model_dump(), **review.model_dump()}
    for key in ("userName", "userUrl", "reviewerId", "avatarUrl"):
        assert key not in combined
    assert "private" not in repr(app)
    assert "private" not in repr(review)


@pytest.mark.parametrize("alias", ["id", "reviewId", "review_id"])
def test_external_id_aliases(alias: str) -> None:
    raw = valid_record()
    del raw["id"]
    raw[alias] = "alias-id"
    _, review = normalize_record(raw, source="src")
    assert review.review_id == "src:alias-id"


@pytest.mark.parametrize("alias", ["appId", "app_id", "bundleId"])
def test_app_id_aliases(alias: str) -> None:
    raw = valid_record()
    del raw["appId"]
    raw[alias] = "alias-app"
    app, _ = normalize_record(raw, source="src")
    assert app.app_id == "alias-app"


@pytest.mark.parametrize("alias", ["appName", "app_name"])
def test_app_name_aliases(alias: str) -> None:
    raw = valid_record()
    del raw["appName"]
    raw[alias] = "Alias App"
    app, _ = normalize_record(raw, source="src")
    assert app.app_name == "Alias App"


@pytest.mark.parametrize("alias", ["score", "rating", "stars"])
def test_rating_aliases(alias: str) -> None:
    raw = valid_record()
    del raw["score"]
    raw[alias] = 5
    _, review = normalize_record(raw, source="src")
    assert review.rating == 5


@pytest.mark.parametrize("alias", ["country", "countryCode", "region"])
def test_country_aliases(alias: str) -> None:
    raw = valid_record()
    del raw["country"]
    raw[alias] = "gb"
    _, review = normalize_record(raw, source="src")
    assert review.country == "GB"


@pytest.mark.parametrize("alias", ["date", "at", "createdAt", "updated"])
def test_created_at_aliases(alias: str) -> None:
    raw = valid_record()
    del raw["date"]
    raw[alias] = "2026-03-04T05:06:07Z"
    _, review = normalize_record(raw, source="src")
    assert review.created_at.hour == 5


@pytest.mark.parametrize("alias", ["title", "reviewTitle"])
def test_title_aliases(alias: str) -> None:
    raw = valid_record()
    del raw["title"]
    raw[alias] = "Alias Title"
    _, review = normalize_record(raw, source="src")
    assert review.title == "Alias Title"


@pytest.mark.parametrize("alias", ["text", "review", "content", "body"])
def test_body_aliases(alias: str) -> None:
    raw = valid_record()
    del raw["text"]
    raw[alias] = "Alias body"
    _, review = normalize_record(raw, source="src")
    assert review.body == "Alias body"


@pytest.mark.parametrize(
    "alias", ["version", "reviewCreatedVersion", "appVersion"]
)
def test_version_aliases(alias: str) -> None:
    raw = valid_record()
    del raw["version"]
    raw[alias] = "9.1"
    _, review = normalize_record(raw, source="src")
    assert review.version == "9.1"


def test_one_bad_record_does_not_abort_the_run() -> None:
    bad = valid_record()
    bad["score"] = 9
    result = normalize_records([bad, valid_record()], source="src")
    assert result.read_count == 2
    assert result.normalized_count == 1
    assert result.rejected_count == 1
