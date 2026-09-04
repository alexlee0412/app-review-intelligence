"""Pure normalization functions for variable Apify review records."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from app.schemas.ingestion import (
    NormalizationResult,
    NormalizedApp,
    NormalizedReview,
    RejectedRecord,
)

EXTERNAL_ID_ALIASES = ("id", "reviewId", "review_id")
APP_ID_ALIASES = ("appId", "app_id", "bundleId")
APP_NAME_ALIASES = ("appName", "app_name")
RATING_ALIASES = ("score", "rating", "stars")
COUNTRY_ALIASES = ("country", "countryCode", "region")
CREATED_AT_ALIASES = ("date", "at", "createdAt", "updated")
TITLE_ALIASES = ("title", "reviewTitle")
BODY_ALIASES = ("text", "review", "content", "body")
VERSION_ALIASES = ("version", "reviewCreatedVersion", "appVersion")


class NormalizationError(ValueError):
    """A single-record validation failure with a stable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _first_value(raw: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        value = raw.get(alias)
        if value is not None:
            return value
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").strip()


def _parse_rating(value: Any) -> int:
    if isinstance(value, bool):
        raise NormalizationError("INVALID_RATING", "rating must be an integer from 1 to 5")
    try:
        numeric = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise NormalizationError(
            "INVALID_RATING", "rating must be an integer from 1 to 5"
        ) from None
    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        raise NormalizationError("INVALID_RATING", "rating must be an integer from 1 to 5")
    rating = int(numeric)
    if rating not in range(1, 6):
        raise NormalizationError("INVALID_RATING", "rating must be an integer from 1 to 5")
    return rating


def _parse_country(value: Any) -> str:
    country = _clean_text(value).upper()
    if re.fullmatch(r"[A-Z]{2}", country, flags=re.ASCII) is None:
        raise NormalizationError(
            "INVALID_COUNTRY", "country must be a two-letter ASCII country code"
        )
    return country


def _parse_created_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith(("Z", "z")):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            raise NormalizationError(
                "INVALID_DATE", "created_at must be a valid ISO-8601 timestamp"
            ) from None
    else:
        raise NormalizationError(
            "INVALID_DATE", "created_at must be a valid ISO-8601 timestamp"
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _review_id(
    source: str,
    external_id: Any,
    app_id: str,
    country: str,
    created_at: datetime,
    body: str,
) -> str:
    external_id_text = _clean_text(external_id)
    candidate = f"{source}:{external_id_text}"
    if external_id_text and len(candidate) <= 64:
        return candidate

    identity = "|".join(
        (app_id, country, created_at.isoformat(), body)
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:32]
    return f"{source[:24]}:{digest}"


def normalize_record(
    raw: dict[str, Any],
    *,
    source: str,
    app_id_override: str | None = None,
) -> tuple[NormalizedApp, NormalizedReview]:
    """Normalize one raw actor record or raise ``NormalizationError``."""
    normalized_source = _clean_text(source)[:64]
    app_id = _clean_text(_first_value(raw, APP_ID_ALIASES))
    if not app_id:
        app_id = _clean_text(app_id_override)
    if not app_id:
        raise NormalizationError("MISSING_APP_ID", "app_id is required")

    rating = _parse_rating(_first_value(raw, RATING_ALIASES))
    country = _parse_country(_first_value(raw, COUNTRY_ALIASES))
    created_at = _parse_created_at(_first_value(raw, CREATED_AT_ALIASES))

    body = _clean_text(_first_value(raw, BODY_ALIASES))
    if not body:
        raise NormalizationError("EMPTY_BODY", "review body must not be empty")

    title = _clean_text(_first_value(raw, TITLE_ALIASES)) or None
    version = _clean_text(_first_value(raw, VERSION_ALIASES))[:32] or None
    app_name = _clean_text(_first_value(raw, APP_NAME_ALIASES)) or app_id
    category = _clean_text(raw.get("category")) or None
    platform = _clean_text(raw.get("platform")) or "ios"

    review_id = _review_id(
        normalized_source,
        _first_value(raw, EXTERNAL_ID_ALIASES),
        app_id,
        country,
        created_at,
        body,
    )

    app = NormalizedApp(
        app_id=app_id,
        app_name=app_name,
        category=category,
        platform=platform,
    )
    review = NormalizedReview(
        review_id=review_id,
        app_id=app_id,
        version=version,
        rating=rating,
        country=country,
        created_at=created_at,
        title=title,
        body=body,
        source=normalized_source,
    )
    return app, review


def normalize_records(
    records: Iterable[dict[str, Any]],
    *,
    source: str,
    app_id_override: str | None = None,
) -> NormalizationResult:
    """Normalize all records while isolating failures to individual records."""
    apps_by_id: dict[str, NormalizedApp] = {}
    reviews: list[NormalizedReview] = []
    rejects: list[RejectedRecord] = []
    reasons: Counter[str] = Counter()
    read_count = 0

    for raw in records:
        read_count += 1
        try:
            app, review = normalize_record(
                raw,
                source=source,
                app_id_override=app_id_override,
            )
        except NormalizationError as exc:
            reasons[exc.reason_code] += 1
            rejects.append(
                RejectedRecord(
                    reason_code=exc.reason_code,
                    message=str(exc),
                    raw=dict(raw),
                )
            )
            continue

        apps_by_id[app.app_id] = app
        reviews.append(review)

    return NormalizationResult(
        apps=list(apps_by_id.values()),
        reviews=reviews,
        rejects=rejects,
        read_count=read_count,
        normalized_count=len(reviews),
        rejected_count=len(rejects),
        rejected_by_reason=dict(sorted(reasons.items())),
    )
