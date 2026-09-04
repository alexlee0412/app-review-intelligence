"""Backfill missing review embeddings in durable batches."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.review import EMBEDDING_DIMENSION, Review
from app.services.embedding_service import (
    EmbeddingDimensionError,
    EmbeddingProvider,
    build_embedding_input,
    build_embedding_provider,
)


@dataclass(frozen=True)
class EmbeddingRunCounts:
    embedded: int
    remaining: int
    skipped: int


def _count_rows(session: Session) -> tuple[int, int]:
    total = session.scalar(select(func.count()).select_from(Review)) or 0
    missing = session.scalar(
        select(func.count()).select_from(Review).where(Review.embedding.is_(None))
    ) or 0
    return total, missing


def backfill_embeddings(
    session_factory: Any,
    provider: EmbeddingProvider,
    *,
    batch_size: int = 128,
    limit: int | None = None,
) -> EmbeddingRunCounts:
    """Fill NULL embeddings and commit each completed batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must not be negative")

    with session_factory() as session:
        initial_total, initial_missing = _count_rows(session)
    embedded = 0

    while limit is None or embedded < limit:
        current_batch_size = batch_size
        if limit is not None:
            current_batch_size = min(current_batch_size, limit - embedded)
        if current_batch_size == 0:
            break

        with session_factory() as session:
            rows = session.execute(
                select(Review.review_id, Review.title, Review.body)
                .where(Review.embedding.is_(None))
                .order_by(Review.review_id)
                .limit(current_batch_size)
            ).all()
            if not rows:
                break

            texts = [build_embedding_input(row.title, row.body) for row in rows]
            vectors = provider.embed_texts(texts)
            if len(vectors) != len(rows):
                raise EmbeddingDimensionError(
                    "Embedding provider returned a different number of vectors than inputs"
                )

            for row, vector in zip(rows, vectors, strict=True):
                if len(vector) != EMBEDDING_DIMENSION:
                    raise EmbeddingDimensionError(
                        f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {len(vector)}"
                    )
                session.execute(
                    update(Review)
                    .where(Review.review_id == row.review_id)
                    .values(embedding=vector)
                )
            session.commit()
            embedded += len(rows)

    with session_factory() as session:
        _, remaining = _count_rows(session)
    return EmbeddingRunCounts(
        embedded=embedded,
        remaining=remaining,
        skipped=initial_total - initial_missing,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fill NULL review embeddings. The reviews table must contain vectors from "
            "exactly one provider/model at a time; mixing vectors makes cosine search "
            "meaningless. Use --reembed-all to clear and refill the table."
        )
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report eligible rows without requesting or writing embeddings",
    )
    parser.add_argument(
        "--reembed-all",
        action="store_true",
        help="NULL every existing vector before refilling with one provider/model",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the destructive --reembed-all reset non-interactively",
    )
    return parser


def _confirm_reembed() -> bool:
    answer = input("Clear every stored embedding before refilling? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


def run(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must not be negative")
    if args.reembed_all and not args.dry_run and not args.yes and not _confirm_reembed():
        print("re-embedding cancelled")
        return 1

    from app.core.db import SessionLocal

    settings = Settings()
    provider = build_embedding_provider(settings)
    print(f"provider={provider.name}")

    if args.dry_run:
        with SessionLocal() as session:
            total, missing = _count_rows(session)
        print(f"embedded=0 remaining={missing} skipped={total - missing}")
        return 0

    if args.reembed_all:
        with SessionLocal.begin() as session:
            session.execute(update(Review).values(embedding=None))

    counts = backfill_embeddings(
        SessionLocal,
        provider,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(
        f"embedded={counts.embedded} remaining={counts.remaining} "
        f"skipped={counts.skipped}"
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args()
    try:
        return run(args)
    except EmbeddingDimensionError as exc:
        print(f"fatal embedding dimension error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
