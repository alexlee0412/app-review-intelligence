"""Service orchestration for evidence-backed semantic review retrieval."""

from __future__ import annotations

import time
import unicodedata
import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import EMBEDDING_DIMENSION, QueryRun
from app.models.query_run import RUN_KIND_SEARCH, RUN_KINDS
from app.repositories.review_search_repository import (
    SearchQueryResult,
    search_review_candidates,
)
from app.schemas.search import (
    AppliedFilters,
    QueryEmbedder,
    QueryTrace,
    ReviewEvidence,
    SearchRequest,
    SearchResponse,
)

CANDIDATE_MULTIPLIER = 2


class QueryEmbeddingError(RuntimeError):
    """Raised when a query embedding cannot be produced safely."""


class InvalidEmbeddingDimensionError(QueryEmbeddingError):
    """Raised when a provider returns a vector with the wrong dimension."""


def _record_stage_latency(session: Session, stage: str, elapsed_ms: float) -> None:
    recorder = getattr(session, "record_stage_latency", None)
    if callable(recorder):
        recorder(stage, elapsed_ms)


def _normalized_filters(request: SearchRequest) -> AppliedFilters:
    return AppliedFilters(
        app_ids=request.app_ids,
        countries=request.countries,
        ratings=request.ratings,
        date_from=request.date_from,
        date_to=request.date_to,
        min_similarity=request.min_similarity,
    )


def _deduplication_key(body: str) -> str:
    without_punctuation = "".join(
        character
        for character in body.lower()
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def _utc_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _evidence_order(evidence: ReviewEvidence) -> tuple[float, float, str]:
    return (-evidence.similarity, -_utc_timestamp(evidence.created_at), evidence.review_id)


def _normalize_evidence(evidence: ReviewEvidence) -> ReviewEvidence:
    similarity = round(min(1.0, max(0.0, float(evidence.similarity))), 6)
    return evidence.model_copy(update={"similarity": similarity})


def _remove_near_duplicates(
    candidates: list[ReviewEvidence],
) -> list[ReviewEvidence]:
    best_by_body: dict[str, ReviewEvidence] = {}
    for raw_candidate in candidates:
        candidate = _normalize_evidence(raw_candidate)
        key = _deduplication_key(candidate.body)
        current = best_by_body.get(key)
        if current is None or _evidence_order(candidate) < _evidence_order(current):
            best_by_body[key] = candidate
    return sorted(best_by_body.values(), key=_evidence_order)


def _embed_query(embedder: QueryEmbedder, query: str) -> list[float]:
    try:
        vector = embedder.embed(query)
    except Exception as exc:
        raise QueryEmbeddingError("query embedding failed") from exc
    if len(vector) != EMBEDDING_DIMENSION:
        raise InvalidEmbeddingDimensionError(
            f"expected embedding dimension {EMBEDDING_DIMENSION}, got {len(vector)}"
        )
    return vector


def _persist_query_run(
    session: Session,
    request: SearchRequest,
    filters: AppliedFilters,
    query_result: SearchQueryResult,
    evidence: list[ReviewEvidence],
    embedder: QueryEmbedder,
    run_kind: str,
) -> UUID:
    query_run_id = uuid.uuid4()
    query_run = QueryRun(
        query_run_id=query_run_id,
        run_kind=run_kind,
        user_query=request.query,
        parsed_intent=None,
        applied_filters=filters.model_dump(mode="json"),
        sql_template=query_result.sql_template,
        result_summary={
            "matched_review_count": query_result.matched_review_count,
            "returned_count": len(evidence),
            "top_similarity": evidence[0].similarity if evidence else None,
            "app_ids_returned": sorted({item.app_id for item in evidence}),
            "embedding_provider": embedder.name,
        },
    )
    session.add(query_run)
    session.commit()
    return query_run_id


def search_reviews(
    session: Session,
    request: SearchRequest,
    embedder: QueryEmbedder,
    *,
    run_kind: str = RUN_KIND_SEARCH,
) -> SearchResponse:
    """Retrieve, deduplicate, trace, and return review evidence.

    ``matched_review_count`` is the pre-deduplication total matching all SQL filters.
    """
    if run_kind not in RUN_KINDS:
        raise ValueError("run_kind must be 'ask' or 'search'")
    filters = _normalized_filters(request)
    embedding_started = time.perf_counter()
    query_vector = _embed_query(embedder, request.query)
    _record_stage_latency(
        session,
        "query_embedding",
        (time.perf_counter() - embedding_started) * 1000,
    )
    retrieval_started = time.perf_counter()
    query_result = search_review_candidates(
        session=session,
        query_vector=query_vector,
        filters=filters,
        top_k=request.top_k,
        candidate_multiplier=CANDIDATE_MULTIPLIER,
    )
    deduplicated = _remove_near_duplicates(query_result.evidence)
    evidence = deduplicated[: request.top_k]
    _record_stage_latency(
        session,
        "retrieval",
        (time.perf_counter() - retrieval_started) * 1000,
    )
    query_run_id = _persist_query_run(
        session=session,
        request=request,
        filters=filters,
        query_result=query_result,
        evidence=evidence,
        embedder=embedder,
        run_kind=run_kind,
    )

    warnings = []
    if not embedder.is_production_grade:
        warnings.append(
            f"The non-semantic development provider ('{embedder.name}') supplied "
            "these embeddings. Similarity scores and result ordering are NOT meaningful."
        )
    if len(deduplicated) < request.top_k <= query_result.matched_review_count:
        warnings.append(
            f"Returned {len(deduplicated)} unique reviews for top_k={request.top_k} "
            f"although {query_result.matched_review_count} reviews matched; "
            "near-duplicate removal exhausted the candidate window."
        )

    return SearchResponse(
        query_run_id=query_run_id,
        matched_review_count=query_result.matched_review_count,
        returned_count=len(evidence),
        warnings=warnings,
        applied_filters=filters,
        evidence=evidence,
        query_trace=QueryTrace(
            sql_template=query_result.sql_template,
            top_k=request.top_k,
            candidate_multiplier=CANDIDATE_MULTIPLIER,
            embedding_provider=embedder.name,
            embedding_dimension=EMBEDDING_DIMENSION,
            similarity_metric="cosine",
        ),
    )
