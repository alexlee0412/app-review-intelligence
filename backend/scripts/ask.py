"""Ask a grounded question about stored app reviews."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.services.ask_service import AskOverrides, answer_question
from app.services.llm_provider import LLMError


def _ratings(value: str) -> list[int]:
    try:
        ratings = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratings must be comma-separated integers") from exc
    if not ratings or any(rating < 1 or rating > 5 for rating in ratings):
        raise argparse.ArgumentTypeError("ratings must be between 1 and 5")
    return list(dict.fromkeys(ratings))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer a grounded question about stored app reviews."
    )
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--app", action="append", dest="apps")
    parser.add_argument("--rating", type=_ratings, dest="ratings")
    return parser


def _session_factory() -> Any:
    from app.core.db import SessionLocal

    return SessionLocal


def _distribution_text(distribution: object) -> str | None:
    if not isinstance(distribution, dict):
        return None
    parts = []
    for rating in range(1, 6):
        value = distribution.get(rating, distribution.get(str(rating)))
        if value is not None:
            parts.append(f"{rating}★ {value}")
    return " · ".join(parts) or None


def _metric_parts(metrics: dict[str, object], *, totals: bool) -> list[str]:
    review_key = "total_reviews" if totals else "review_count"
    matched_key = "total_matched" if totals else "matched_count"
    average_key = "overall_avg_rating" if totals else "avg_rating"
    parts: list[str] = []
    if metrics.get(review_key) is not None:
        parts.append(f"Reviews analyzed: {metrics[review_key]}")
    if metrics.get(matched_key) is not None:
        parts.append(f"Matched reviews: {metrics[matched_key]}")
    average = metrics.get(average_key)
    if average is not None:
        parts.append(f"Average rating: {average:.2f}")
    return parts


def _print_metrics(metrics: dict[str, object]) -> None:
    print("Metrics:")
    totals = metrics.get("totals")
    if isinstance(totals, dict):
        for part in _metric_parts(totals, totals=True):
            print(part)
        distribution = _distribution_text(totals.get("rating_distribution"))
        if distribution is not None:
            print(f"Rating distribution: {distribution}")

    apps = metrics.get("apps")
    app_metrics = (
        [item for item in apps if isinstance(item, dict)]
        if isinstance(apps, list)
        else []
    )
    if len(app_metrics) > 1 or (app_metrics and not isinstance(totals, dict)):
        for app in app_metrics:
            name = app.get("app_name") or app.get("app_id") or "App"
            parts = _metric_parts(app, totals=False)
            if parts:
                print(f"{name}: {' · '.join(parts)}")
            distribution = _distribution_text(app.get("rating_distribution"))
            if distribution is not None:
                print(f"  Rating distribution: {distribution}")


def _print_human(response: object) -> None:
    print("Answer:")
    print(response.answer)
    if response.metrics:
        _print_metrics(response.metrics)
    print("Findings:")
    if response.findings:
        for finding in response.findings:
            citations = " ".join(f"[{item}]" for item in finding.evidence_ids)
            print(f"- [{finding.kind}] {finding.claim} {citations}".rstrip())
    else:
        print("- Nothing could be grounded.")
    print("Evidence:")
    for item in response.evidence:
        review = item.review
        print(
            f"[{item.evidence_id}] {review.app_name}, {review.rating}★, "
            f"{review.created_at.date().isoformat()}"
        )
        print(f"  {json.dumps(item.excerpt, ensure_ascii=False)}")
    print("Limitations:")
    for limitation in [*response.limitations, *response.warnings]:
        print(f"- {limitation}")
    print("Trace:")
    print(json.dumps(response.trace.model_dump(mode="json"), ensure_ascii=False))


def run(args: argparse.Namespace) -> int:
    overrides = AskOverrides(
        top_k=args.top_k,
        apps=args.apps,
        ratings=args.ratings,
    )
    with _session_factory()() as session:
        response = answer_question(session, args.question, overrides=overrides)
    if args.as_json:
        print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False))
    else:
        _print_human(response)
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        # The class name distinguishes a provider timeout from a database or
        # validation failure. Only our own errors expose their message, since an
        # arbitrary exception can embed a connection string.
        detail = str(exc) if isinstance(exc, LLMError) else ""
        suffix = f": {detail}" if detail else ""
        print(
            f"fatal: unable to answer question ({type(exc).__name__}){suffix}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
