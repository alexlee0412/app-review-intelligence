"""Normalize Apify review data and upsert it into PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services.apify_source import fetch_dataset, load_from_file
from app.services.normalization import normalize_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest normalized Apify reviews into the configured database."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--file", type=Path, help="JSON array or JSONL input")
    source_group.add_argument("--dataset-id", help="Apify dataset identifier")
    parser.add_argument("--source", required=True, help="Source label stored per review")
    parser.add_argument("--app-id", help="Fallback app identifier")
    parser.add_argument("--limit", type=int, help="Maximum source records to read")
    parser.add_argument("--dry-run", action="store_true", help="Normalize without writing")
    parser.add_argument(
        "--rejects-dir",
        type=Path,
        default=Path("../data/rejects"),
        help="Relative directory for timestamped reject JSONL files",
    )
    return parser


def _limited(
    records: Iterable[dict[str, Any]], limit: int | None
) -> Iterable[dict[str, Any]]:
    if limit is None:
        return records
    if limit < 0:
        raise ValueError("--limit must not be negative")
    return islice(records, limit)


def _write_rejects(rejects_dir: Path, rejects: list[Any]) -> Path | None:
    if not rejects:
        return None
    rejects_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = rejects_dir / f"{timestamp}.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for reject in rejects:
            handle.write(json.dumps(reject.model_dump(), ensure_ascii=False) + "\n")
    return path


def _summary(
    *,
    read: int,
    normalized: int,
    rejected: int,
    rejected_by_reason: dict[str, int],
    inserted: int,
    updated: int,
) -> str:
    reasons = ", ".join(
        f"{reason}={count}" for reason, count in sorted(rejected_by_reason.items())
    ) or "none"
    return (
        f"read={read} normalized={normalized} rejected={rejected} "
        f"rejected_by_reason=[{reasons}] inserted={inserted} updated={updated}"
    )


def run(args: argparse.Namespace) -> int:
    settings: Settings | None = None
    if args.file is not None:
        records = load_from_file(args.file)
    else:
        settings = Settings()
        if settings.apify_api_token is None:
            raise ValueError(
                "APP_APIFY_API_TOKEN is required when reading an Apify dataset"
            )
        records = fetch_dataset(
            args.dataset_id,
            settings.apify_api_token.get_secret_value(),
        )

    result = normalize_records(
        _limited(records, args.limit),
        source=args.source,
        app_id_override=args.app_id,
    )
    reject_path = _write_rejects(args.rejects_dir, result.rejects)

    inserted = 0
    updated = 0
    if not args.dry_run:
        from app.core.db import SessionLocal
        from app.repositories.review_write_repository import (
            upsert_normalized_records,
        )

        with SessionLocal.begin() as session:
            counts = upsert_normalized_records(
                session,
                result.apps,
                result.reviews,
            )
        inserted = counts.inserted
        updated = counts.updated

    print(
        _summary(
            read=result.read_count,
            normalized=result.normalized_count,
            rejected=result.rejected_count,
            rejected_by_reason=result.rejected_by_reason,
            inserted=inserted,
            updated=updated,
        )
    )
    if reject_path is not None:
        print(f"rejects={reject_path}")
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
