"""Build citation-ready evidence bundles from validated query plans."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.schemas.answer import (
    MIN_EVIDENCE_FOR_SYNTHESIS,
    AppAggregate,
    EvidenceBundle,
    EvidenceItem,
    RetrievalTrace,
    evidence_id_for,
)
from app.schemas.query_plan import QueryPlan
from app.schemas.search import AppliedFilters, QueryEmbedder, SearchRequest
from app.services.review_search import search_reviews

MAX_EXCERPT_CHARACTERS = 500


def _filters_for(plan: QueryPlan) -> AppliedFilters:
    return AppliedFilters(
        app_ids=plan.app_ids,
        countries=plan.countries,
        ratings=plan.ratings,
        date_from=plan.date_from,
        date_to=plan.date_to,
        min_similarity=None,
    )


def _thin_evidence_limitation(count: int) -> str:
    return (
        f"Only {count} evidence items were available; at least "
        f"{MIN_EVIDENCE_FOR_SYNTHESIS} are required for answer synthesis."
    )


def build_evidence(
    session: Session,
    plan: QueryPlan,
    *,
    embedder: QueryEmbedder,
    aggregates: list[AppAggregate] | None = None,
) -> EvidenceBundle:
    """Retrieve reviews once and package deterministic, verbatim evidence."""
    applied_filters = _filters_for(plan)
    evidence: list[EvidenceItem] = []
    limitations: list[str] = []
    total_candidates = 0
    trace = RetrievalTrace(
        semantic_query=plan.semantic_query,
        top_k=plan.top_k,
        total_candidates=0,
        returned_evidence=0,
    )

    if plan.needs_semantic_search:
        search_response = search_reviews(
            session,
            SearchRequest(
                query=plan.semantic_query,
                app_ids=plan.app_ids,
                countries=plan.countries,
                ratings=plan.ratings,
                date_from=plan.date_from,
                date_to=plan.date_to,
                top_k=plan.top_k,
            ),
            embedder,
        )
        applied_filters = search_response.applied_filters
        total_candidates = search_response.matched_review_count
        limitations.extend(search_response.warnings)
        evidence = [
            EvidenceItem(
                evidence_id=evidence_id_for(position),
                review=review,
                excerpt=review.body[:MAX_EXCERPT_CHARACTERS],
            )
            for position, review in enumerate(search_response.evidence, start=1)
        ]
        trace = RetrievalTrace(
            semantic_query=plan.semantic_query,
            top_k=plan.top_k,
            total_candidates=total_candidates,
            returned_evidence=len(evidence),
            embedding_provider=search_response.query_trace.embedding_provider,
            embedding_is_production_grade=embedder.is_production_grade,
            similarity_metric=search_response.query_trace.similarity_metric,
            sql_template=search_response.query_trace.sql_template,
        )

    if len(evidence) < MIN_EVIDENCE_FOR_SYNTHESIS:
        limitations.append(_thin_evidence_limitation(len(evidence)))

    return EvidenceBundle(
        query_plan=plan,
        applied_filters=applied_filters,
        aggregates=aggregates or [],
        evidence=evidence,
        total_candidates=total_candidates,
        limitations=list(dict.fromkeys(limitations)),
        retrieval_trace=trace,
    )
