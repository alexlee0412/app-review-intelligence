from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest

from app.schemas.answer import (
    AppAggregate,
    EvidenceBundle,
    EvidenceItem,
    Finding,
    RetrievalTrace,
    SynthesisOutput,
)
from app.schemas.query_plan import Intent, QueryPlan
from app.schemas.search import AppliedFilters, ReviewEvidence
from app.services.answer_synthesizer import (
    _SYSTEM_PROMPT,
    _payload,
    _validate_findings,
)
from app.services.metric_formatting import (
    UNAVAILABLE,
    format_average,
    format_bundle_metrics,
    format_count,
    format_date,
    format_date_range,
    format_delta,
    format_percentage,
)


def _bundle() -> EvidenceBundle:
    oldest = datetime(2026, 1, 2, 4, 5, tzinfo=timezone.utc)
    newest = datetime(2026, 9, 3, 6, 7, tzinfo=timezone.utc)
    distribution = {1: 600, 2: 300, 3: 200, 4: 100, 5: 34}
    average = 2.2857142857142856
    plan = QueryPlan(
        intent=Intent.APP_COMPARISON,
        semantic_query="subscription problems",
        needs_aggregation=True,
    )
    aggregates = [
        AppAggregate(
            app_id="app-one",
            app_name="App One",
            review_count=1234,
            matched_count=10,
            avg_rating=average,
            rating_distribution=distribution,
            oldest_review_at=oldest,
            newest_review_at=newest,
        ),
        AppAggregate(
            app_id="app-two",
            app_name="App Two",
            review_count=0,
            matched_count=0,
        ),
    ]
    return EvidenceBundle(
        query_plan=plan,
        applied_filters=AppliedFilters(countries=["US"]),
        aggregates=aggregates,
        totals={
            "total_reviews": 1234,
            "total_matched": 10,
            "apps_with_matches": 1,
            "overall_avg_rating": average,
            "rating_distribution": distribution,
        },
        evidence=[
            EvidenceItem(
                evidence_id="E1",
                review=ReviewEvidence(
                    review_id="review-one",
                    app_id="app-one",
                    app_name="App One",
                    version="1.0",
                    rating=2,
                    country="US",
                    created_at=newest,
                    title=None,
                    body="Stored review body.",
                    similarity=0.9,
                ),
                excerpt="Stored review body.",
            )
        ],
        total_candidates=1,
        retrieval_trace=RetrievalTrace(
            semantic_query=plan.semantic_query,
            top_k=1,
            total_candidates=1,
            returned_evidence=1,
        ),
    )


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _strings(nested)]
    return []


def _identity_aggregates() -> list[AppAggregate]:
    return [
        AppAggregate(
            app_id="1577705074",
            app_name="B612",
            review_count=10,
            matched_count=4,
            avg_rating=4.2,
        ),
        AppAggregate(
            app_id="app-two",
            app_name="Second App",
            review_count=20,
            matched_count=5,
            avg_rating=2.1,
        ),
        AppAggregate(
            app_id="app-three",
            app_name="Third App",
            review_count=30,
            matched_count=6,
            avg_rating=3.3,
        ),
    ]


def test_count_formatting_handles_inflection_and_large_values() -> None:
    assert format_count(1, singular="review") == "1 review"
    assert format_count(10, singular="review") == "10 reviews"
    assert format_count(12345, singular="review") == "12,345 reviews"


def test_average_formatting_uses_two_decimal_places() -> None:
    assert format_average(2.2857142857142856) == "2.29"
    assert format_average(2.2) == "2.20"


def test_percentage_formatting_uses_one_decimal_place() -> None:
    assert format_percentage(42.94) == "42.9%"
    assert format_percentage(80.0) == "80.0%"


def test_delta_formatting_uses_explicit_signs() -> None:
    assert format_delta(3) == "+3"
    assert format_delta(-2) == "-2"


def test_date_and_date_range_formatting_is_stable() -> None:
    timestamp = datetime(2026, 9, 3, 18, 30, tzinfo=timezone.utc)
    assert format_date(timestamp) == "2026-09-03"
    assert format_date_range(date(2026, 6, 1), timestamp) == (
        "2026-06-01 to 2026-09-03"
    )


@pytest.mark.parametrize(
    "formatted",
    [
        format_count(None),
        format_average(None),
        format_percentage(None),
        format_delta(None),
        format_date(None),
        format_date_range(None, date(2026, 1, 1)),
        format_date_range(date(2026, 1, 1), None),
    ],
)
def test_none_always_uses_the_unavailable_sentinel(formatted: str) -> None:
    assert formatted == UNAVAILABLE


def test_formatting_is_pure_and_stable() -> None:
    bundle = _bundle()
    first = format_bundle_metrics(bundle.totals, bundle.aggregates)
    second = format_bundle_metrics(bundle.totals, bundle.aggregates)

    assert first == second
    assert bundle.totals["overall_avg_rating"] == 2.2857142857142856
    assert bundle.aggregates[0].avg_rating == 2.2857142857142856


def test_payload_keeps_full_precision_raw_values_alongside_display_values() -> None:
    payload = _payload(_bundle())

    assert payload["totals"]["overall_avg_rating"] == 2.2857142857142856
    assert payload["aggregates"][0]["avg_rating"] == 2.2857142857142856
    assert payload["formatted"]["totals"]["overall_avg_rating"] == "2.29"
    assert payload["formatted"]["totals"]["rating_distribution"]["one_star"] == (
        "600 one-star reviews"
    )
    assert payload["formatted"]["aggregates"][0]["avg_rating"] == "2.29"
    assert set(payload["formatted"]["totals"]) == set(payload["totals"])
    assert set(payload["formatted"]["aggregates"][0]) == {
        "app_id",
        "app_name",
        "review_count",
        "matched_count",
        "avg_rating",
        "rating_distribution",
        "oldest_review_at",
        "newest_review_at",
    }


def test_every_emitted_display_string_survives_computed_validation() -> None:
    bundle = _bundle()
    displays = _strings(_payload(bundle)["formatted"])
    output = SynthesisOutput(
        answer="The computed display values are listed in the findings.",
        findings=[
            Finding(
                claim=f"The supplied display value is {display}.",
                evidence_ids=["E1"],
                kind="computed",
            )
            for display in displays
        ],
    )

    validated, limitations, _ = _validate_findings(output, bundle)

    assert len(validated.findings) == len(displays)
    assert limitations == []


def test_formatted_aggregates_bind_identity_to_distinct_figures() -> None:
    aggregates = _identity_aggregates()[:2]
    formatted = format_bundle_metrics({}, aggregates)["aggregates"]

    assert formatted == [
        {
            "app_id": "1577705074",
            "app_name": "B612",
            "review_count": "10 reviews",
            "matched_count": "4 matching reviews",
            "avg_rating": "4.20",
            "rating_distribution": {},
            "oldest_review_at": UNAVAILABLE,
            "newest_review_at": UNAVAILABLE,
        },
        {
            "app_id": "app-two",
            "app_name": "Second App",
            "review_count": "20 reviews",
            "matched_count": "5 matching reviews",
            "avg_rating": "2.10",
            "rating_distribution": {},
            "oldest_review_at": UNAVAILABLE,
            "newest_review_at": UNAVAILABLE,
        },
    ]


def test_reordering_aggregates_preserves_identity_figure_binding() -> None:
    aggregates = _identity_aggregates()[:2]
    original = format_bundle_metrics({}, aggregates)["aggregates"]
    reordered = format_bundle_metrics({}, list(reversed(aggregates)))["aggregates"]

    original_by_id = {item["app_id"]: item for item in original}
    reordered_by_id = {item["app_id"]: item for item in reordered}
    assert reordered_by_id == original_by_id
    assert [item["app_id"] for item in reordered] == ["app-two", "1577705074"]


def test_every_formatted_identity_matches_raw_aggregate() -> None:
    aggregates = _identity_aggregates()
    raw = [aggregate.model_dump(mode="json") for aggregate in aggregates]
    formatted = format_bundle_metrics({}, aggregates)["aggregates"]

    assert len(formatted) == 3
    for raw_entry, formatted_entry in zip(raw, formatted, strict=True):
        assert formatted_entry["app_id"] == raw_entry["app_id"]
        assert formatted_entry["app_name"] == raw_entry["app_name"]


def test_digit_bearing_formatted_identity_does_not_drop_supported_figure() -> None:
    bundle = _bundle().model_copy(
        update={"aggregates": _identity_aggregates()[:2]}
    )
    formatted = format_bundle_metrics({}, bundle.aggregates)["aggregates"][0]
    output = SynthesisOutput(
        answer="The validated finding reports the computed average.",
        findings=[
            Finding(
                claim=(
                    f"{formatted['app_name']} has an average rating of "
                    f"{formatted['avg_rating']}."
                ),
                evidence_ids=["E1"],
                kind="computed",
            )
        ],
    )

    validated, limitations, _ = _validate_findings(output, bundle)

    assert validated.findings == output.findings
    assert limitations == []


def test_single_app_formatted_metrics_keep_existing_values() -> None:
    aggregate = _identity_aggregates()[0]
    formatted = format_bundle_metrics({}, [aggregate])["aggregates"]

    assert len(formatted) == 1
    assert formatted[0]["review_count"] == "10 reviews"
    assert formatted[0]["matched_count"] == "4 matching reviews"
    assert formatted[0]["avg_rating"] == "4.20"


def test_prompt_requires_verbatim_display_values_without_derivation() -> None:
    assert "formatted display string verbatim" in _SYSTEM_PROMPT
    assert (
        "Never round, reformat, recompute, convert units, or derive" in _SYSTEM_PROMPT
    )
