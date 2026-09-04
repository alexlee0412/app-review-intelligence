"""Database access for exact cosine review retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.models import EMBEDDING_DIMENSION, App, Review
from app.schemas.search import AppliedFilters, MAX_TOP_K, ReviewEvidence

MAX_CANDIDATE_LIMIT = MAX_TOP_K * 2


@dataclass(frozen=True)
class SearchQueryResult:
    """Candidate evidence and trace data returned by one search statement."""

    evidence: list[ReviewEvidence]
    matched_review_count: int
    sql_template: str


def normalize_similarity(distance: float) -> float:
    """Convert cosine distance to a response-safe similarity value."""
    similarity = 1.0 - float(distance)
    if not isfinite(similarity):
        similarity = 0.0
    return round(min(1.0, max(0.0, similarity)), 6)


def search_review_candidates(
    session: Session,
    query_vector: list[float],
    filters: AppliedFilters,
    top_k: int,
    candidate_multiplier: int,
) -> SearchQueryResult:
    """Run one exact-cosine query and return deterministic candidates."""
    query_embedding = bindparam(
        "query_embedding",
        value=query_vector,
        type_=Vector(EMBEDDING_DIMENSION),
    )
    distance = Review.embedding.cosine_distance(query_embedding)

    statement = (
        select(
            Review.review_id,
            Review.app_id,
            App.app_name,
            Review.version,
            Review.rating,
            Review.country,
            Review.created_at,
            Review.title,
            Review.body,
            distance.label("distance"),
            func.count().over().label("matched_review_count"),
        )
        .join(App, App.app_id == Review.app_id)
        .where(Review.embedding.is_not(None))
    )

    if filters.app_ids is not None:
        statement = statement.where(Review.app_id.in_(filters.app_ids))
    if filters.countries is not None:
        statement = statement.where(Review.country.in_(filters.countries))
    if filters.ratings is not None:
        statement = statement.where(Review.rating.in_(filters.ratings))
    if filters.date_from is not None:
        statement = statement.where(Review.created_at >= filters.date_from)
    if filters.date_to is not None:
        statement = statement.where(Review.created_at < filters.date_to)
    if filters.min_similarity is not None:
        statement = statement.where(distance <= 1.0 - filters.min_similarity)

    candidate_limit = min(top_k * candidate_multiplier, MAX_CANDIDATE_LIMIT)
    statement = statement.order_by(
        distance.asc(),
        Review.created_at.desc(),
        Review.review_id.asc(),
    ).limit(candidate_limit)

    sql_template = str(statement.compile(dialect=postgresql.dialect()))
    rows = session.execute(statement).all()
    if not rows:
        return SearchQueryResult(
            evidence=[], matched_review_count=0, sql_template=sql_template
        )

    evidence = []
    for row in rows:
        values = row._mapping
        evidence.append(
            ReviewEvidence(
                review_id=values["review_id"],
                app_id=values["app_id"],
                app_name=values["app_name"],
                version=values["version"],
                rating=values["rating"],
                country=values["country"],
                created_at=values["created_at"],
                title=values["title"],
                body=values["body"],
                similarity=normalize_similarity(values["distance"]),
            )
        )

    return SearchQueryResult(
        evidence=evidence,
        matched_review_count=int(rows[0]._mapping["matched_review_count"]),
        sql_template=sql_template,
    )
