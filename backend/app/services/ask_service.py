"""Canonical orchestration for grounded review question answering."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models import App, QueryRun
from app.models.query_run import RUN_KIND_ASK, SCHEMA_VERSION
from app.schemas.answer import (
    AnswerResponse,
    AnswerTrace,
    AppAggregate,
    EvidenceBundle,
    SynthesisOutput,
)
from app.schemas.query_plan import (
    NON_ANSWERING_INTENTS,
    AppCatalogEntry,
    Intent,
    PlannerResult,
    QueryPlan,
)
from app.schemas.search import QueryEmbedder
from app.services.answer_synthesizer import (
    PROMPT_VERSION as SYNTHESIZER_PROMPT_VERSION,
    synthesize_answer,
)
from app.services.evidence_builder import build_evidence


class Planner(Protocol):
    def __call__(
        self,
        question: str,
        app_catalog: list[AppCatalogEntry],
        *,
        client: Any,
        settings: Settings,
    ) -> PlannerResult: ...


AggregateBuilder = Callable[[Session, QueryPlan, set[str]], list[AppAggregate]]
TotalsBuilder = Callable[[list[AppAggregate]], dict[str, Any]]


@dataclass(frozen=True)
class AskOverrides:
    """Optional caller-supplied restrictions applied after planning."""

    top_k: int | None = None
    apps: list[str] | None = None
    ratings: list[int] | None = None


class _TrackingSession:
    """Delegate retrieval work while retaining its trace row for finalization."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self.query_run: QueryRun | None = None
        self.query_run_id: uuid.UUID | None = None
        self.stage_latency_ms: dict[str, float] = {}

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._session.execute(*args, **kwargs)

    def add(self, instance: object) -> None:
        if isinstance(instance, QueryRun):
            self.query_run = instance
            self.query_run_id = instance.query_run_id
        self._session.add(instance)

    def commit(self) -> None:
        self._session.commit()

    def record_stage_latency(self, stage: str, elapsed_ms: float) -> None:
        self.stage_latency_ms[stage] = elapsed_ms

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


def _build_client(settings: Settings) -> Any:
    from app.services.llm_provider import build_llm_client

    return build_llm_client(settings)


def _build_embedder(settings: Settings) -> QueryEmbedder:
    from app.services.embedding_service import build_query_embedder

    return build_query_embedder(settings)


def _default_planner(
    question: str,
    app_catalog: list[AppCatalogEntry],
    *,
    client: Any,
    settings: Settings,
) -> PlannerResult:
    from app.services.query_planner import plan_question

    return plan_question(
        question,
        app_catalog,
        client=client,
        settings=settings,
    )


def _default_aggregate_builder(
    session: Session, plan: QueryPlan, matched_review_ids: set[str]
) -> list[AppAggregate]:
    from app.repositories.review_analytics_repository import aggregate_reviews_by_app

    return aggregate_reviews_by_app(
        session,
        app_ids=plan.app_ids,
        countries=plan.countries,
        ratings=plan.ratings,
        date_from=plan.date_from,
        date_to=plan.date_to,
        matched_review_ids=matched_review_ids,
    )


def _default_totals_builder(aggregates: list[AppAggregate]) -> dict[str, Any]:
    from app.services.analytics import build_analytics_totals

    return build_analytics_totals(aggregates)


def _load_app_catalog(session: Session) -> list[AppCatalogEntry]:
    rows = session.execute(
        select(App.app_id, App.app_name).order_by(App.app_id)
    ).all()
    return [
        AppCatalogEntry(app_id=row.app_id, app_name=row.app_name) for row in rows
    ]


def _resolve_override_apps(
    requested: list[str], catalog: list[AppCatalogEntry]
) -> tuple[list[str], list[str]]:
    by_id = {entry.app_id: entry.app_id for entry in catalog}
    by_name: dict[str, list[str]] = {}
    for entry in catalog:
        by_name.setdefault(entry.app_name.casefold(), []).append(entry.app_id)

    resolved: list[str] = []
    limitations: list[str] = []
    for value in requested:
        normalized = value.strip()
        matches = [by_id[normalized]] if normalized in by_id else by_name.get(
            normalized.casefold(), []
        )
        if not matches:
            limitations.append(
                f"Ignored unknown app name or identifier {value!r}; it is not in the "
                "dataset."
            )
        resolved.extend(matches)
    return list(dict.fromkeys(resolved)), limitations


def _apply_overrides(
    plan: QueryPlan,
    catalog: list[AppCatalogEntry],
    overrides: AskOverrides | None,
) -> tuple[QueryPlan, list[str], bool]:
    if overrides is None:
        return plan, [], False

    values = plan.model_dump(mode="python")
    limitations: list[str] = []
    empty_app_scope = False
    if overrides.top_k is not None:
        values["top_k"] = overrides.top_k
    if overrides.ratings is not None:
        values["ratings"] = overrides.ratings
    if overrides.apps is not None:
        resolved, app_limitations = _resolve_override_apps(overrides.apps, catalog)
        limitations.extend(app_limitations)
        empty_app_scope = not resolved
        values["app_ids"] = resolved or None
    return QueryPlan.model_validate(values), limitations, empty_app_scope


def _metrics(
    aggregates: list[AppAggregate], totals: dict[str, Any]
) -> dict[str, Any]:
    if not aggregates:
        return {}
    return {
        "totals": totals,
        "apps": [aggregate.model_dump(mode="json") for aggregate in aggregates],
    }


def _model_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "provider": usage.get("provider"),
        "model": usage.get("model"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "cost": None,
    }


def _retrieval_outcomes(
    bundle: EvidenceBundle, synthesis: SynthesisOutput
) -> dict[str, Any]:
    returned_ids = [item.evidence_id for item in bundle.evidence]
    valid_ids = set(returned_ids)
    cited = {
        evidence_id
        for finding in synthesis.findings
        for evidence_id in finding.evidence_ids
        if evidence_id in valid_ids
    }
    cited.update(
        match.group(1)
        for match in re.finditer(r"(?<!\w)\[?(E\d+)\]?(?!\w)", synthesis.answer)
        if match.group(1) in valid_ids
    )
    cited_ids = [evidence_id for evidence_id in returned_ids if evidence_id in cited]
    returned_count = len(returned_ids)
    return {
        "returned_evidence_ids": returned_ids,
        "cited_evidence_ids": cited_ids,
        "returned_count": returned_count,
        "cited_count": len(cited_ids),
        "cited_ratio": len(cited_ids) / returned_count if returned_count else None,
    }


def _run_versions() -> dict[str, Any]:
    from app.services.query_planner import PROMPT_VERSION as PLANNER_PROMPT_VERSION

    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": {
            "planner": PLANNER_PROMPT_VERSION,
            "synthesizer": SYNTHESIZER_PROMPT_VERSION,
        },
        "taxonomy_version": None,
    }


def _insufficient_answer(plan: QueryPlan, evidence_count: int) -> str:
    if plan.intent is Intent.TREND_ANALYSIS:
        return "The available dataset does not support reliable trend analysis."
    if plan.intent is Intent.UNSUPPORTED:
        return "This question is outside the supported review-analysis scope."
    return (
        f"Insufficient evidence: only {evidence_count} supporting reviews were found, "
        "so no synthesized conclusion was produced."
    )


def _provider_warnings(bundle: EvidenceBundle, client: Any) -> list[str]:
    warnings: list[str] = []
    if not client.is_production_grade:
        warnings.append(
            f"The non-production answer provider ('{client.name}') was used; generated "
            "language is not production-grade."
        )
    if bundle.retrieval_trace.embedding_is_production_grade is False:
        provider = bundle.retrieval_trace.embedding_provider or "unknown"
        warnings.append(
            f"The non-semantic development embedding provider ('{provider}') was used; "
            "retrieval scores and ordering are not meaningful."
        )
    return warnings


def _persist_answer_run(
    session: Session,
    tracking_session: _TrackingSession,
    *,
    question: str,
    plan: QueryPlan,
    bundle: EvidenceBundle,
    response_fields: dict[str, Any],
    stage_latency_ms: dict[str, float | None],
    model_usage: dict[str, Any],
    grounding_outcomes: dict[str, Any] | None,
    retrieval_outcomes: dict[str, Any],
    total_started: float,
) -> uuid.UUID:
    query_run_id = tracking_session.query_run_id or uuid.uuid4()
    query_run = tracking_session.query_run or QueryRun(query_run_id=query_run_id)
    trace: AnswerTrace = response_fields["trace"]
    query_run.run_kind = RUN_KIND_ASK
    query_run.user_query = question
    query_run.parsed_intent = plan.model_dump(mode="json")
    query_run.applied_filters = bundle.applied_filters.model_dump(mode="json")
    query_run.sql_template = bundle.retrieval_trace.sql_template
    query_run.result_summary = {
        "total_candidates": bundle.total_candidates,
        "evidence_count": len(bundle.evidence),
        "finding_count": len(response_fields["findings"]),
        "aggregate_count": len(bundle.aggregates),
        "planner_model": trace.planner_model,
        "synthesizer_model": trace.synthesizer_model,
        "llm_provider": trace.llm_provider,
        "embedding_provider": trace.embedding_provider,
        "synthesis_skipped": trace.synthesis_skipped,
    }
    query_run.stage_latency_ms = stage_latency_ms
    query_run.model_usage = model_usage
    query_run.grounding_outcomes = grounding_outcomes
    query_run.retrieval_outcomes = retrieval_outcomes
    query_run.run_versions = _run_versions()
    stage_latency_ms["total"] = (time.perf_counter() - total_started) * 1000
    session.add(query_run)
    session.commit()
    return query_run_id


def answer_question(
    session: Session,
    question: str,
    *,
    overrides: AskOverrides | None = None,
    planner: Planner | None = None,
    aggregate_builder: AggregateBuilder | None = None,
    totals_builder: TotalsBuilder | None = None,
    embedder: QueryEmbedder | None = None,
    client: Any | None = None,
    settings: Settings | None = None,
) -> AnswerResponse:
    """Plan, execute, validate, trace, and persist one grounded answer."""
    total_started = time.perf_counter()
    question = question.strip()
    if not question:
        raise ValueError("question must not be blank")

    resolved_settings = settings or get_settings()
    catalog = _load_app_catalog(session)
    resolved_client = client or _build_client(resolved_settings)
    resolved_planner = planner or _default_planner
    planner_started = time.perf_counter()
    planner_result = resolved_planner(
        question,
        catalog,
        client=resolved_client,
        settings=resolved_settings,
    )
    planner_latency_ms = (time.perf_counter() - planner_started) * 1000
    plan, override_limitations, empty_override_scope = _apply_overrides(
        planner_result.plan, catalog, overrides
    )
    planner_dropped_all_apps = (
        plan.app_ids is None and bool(planner_result.dropped_app_ids)
    )
    had_planned_apps = plan.app_ids is not None
    plan, restriction_limitations, _ = plan.restrict_apps_to(
        {entry.app_id for entry in catalog}
    )
    empty_app_scope = (
        empty_override_scope
        or planner_dropped_all_apps
        or (had_planned_apps and plan.app_ids is None)
    )
    if empty_app_scope:
        plan = plan.model_copy(
            update={"needs_semantic_search": False, "needs_aggregation": False}
        )

    resolved_embedder = embedder or _build_embedder(resolved_settings)
    tracking_session = _TrackingSession(session)
    bundle = build_evidence(
        tracking_session,  # type: ignore[arg-type]
        plan,
        embedder=resolved_embedder,
    )
    aggregates: list[AppAggregate] = []
    totals: dict[str, Any] = {}
    analytics_latency_ms: float | None = None
    if plan.needs_aggregation:
        analytics_started = time.perf_counter()
        resolved_aggregate_builder = aggregate_builder or _default_aggregate_builder
        aggregates = TypeAdapter(list[AppAggregate]).validate_python(
            resolved_aggregate_builder(
                session,
                plan,
                {item.review.review_id for item in bundle.evidence},
            )
        )
        resolved_totals_builder = totals_builder or _default_totals_builder
        totals = resolved_totals_builder(aggregates)
        analytics_latency_ms = (time.perf_counter() - analytics_started) * 1000
    limitations = list(
        dict.fromkeys(
            planner_result.limitations
            + override_limitations
            + restriction_limitations
            + bundle.limitations
        )
    )
    bundle = bundle.model_copy(
        update={
            "aggregates": aggregates,
            "totals": totals,
            "limitations": limitations,
        }
    )

    synthesis_skipped = (
        not bundle.has_sufficient_evidence or plan.intent in NON_ANSWERING_INTENTS
    )
    synthesis_limitations: list[str] = []
    grounding_outcomes: dict[str, Any] | None = None
    synthesizer_usage: dict[str, Any] | None = None
    synthesis_latency_ms: float | None = None
    validation_latency_ms: float | None = None
    if synthesis_skipped:
        synthesis = SynthesisOutput(
            answer=_insufficient_answer(plan, len(bundle.evidence)), findings=[]
        )
    else:
        synthesis_result = synthesize_answer(
            bundle,
            client=resolved_client,
            model=resolved_settings.synthesizer_model,
            max_output_tokens=resolved_settings.llm_max_output_tokens,
        )
        synthesis = synthesis_result.output
        synthesis_limitations = synthesis_result.limitations
        grounding_outcomes = synthesis_result.grounding_outcomes
        synthesizer_usage = synthesis_result.model_usage
        synthesis_latency_ms = synthesis_result.synthesis_latency_ms
        validation_latency_ms = synthesis_result.validation_latency_ms

    limitations = list(dict.fromkeys(limitations + synthesis_limitations))
    trace = AnswerTrace(
        intent=plan.intent.value,
        applied_filters=bundle.applied_filters,
        semantic_query=plan.semantic_query,
        total_candidates=bundle.total_candidates,
        evidence_count=len(bundle.evidence),
        aggregates_computed=list(plan.requested_metrics) if aggregates else [],
        planner_model=planner_result.model,
        synthesizer_model=(
            None if synthesis_skipped else resolved_settings.synthesizer_model
        ),
        llm_provider=resolved_client.name,
        llm_is_production_grade=resolved_client.is_production_grade,
        embedding_provider=bundle.retrieval_trace.embedding_provider,
        embedding_is_production_grade=(
            bundle.retrieval_trace.embedding_is_production_grade
        ),
        synthesis_skipped=synthesis_skipped,
    )
    response_fields = {
        "question": question,
        "answer": synthesis.answer,
        "findings": synthesis.findings,
        "metrics": _metrics(aggregates, totals),
        "evidence": bundle.evidence,
        "limitations": limitations,
        "warnings": _provider_warnings(bundle, resolved_client),
        "trace": trace,
    }
    stage_latency_ms: dict[str, float | None] = {
        "planner": planner_latency_ms,
        "query_embedding": tracking_session.stage_latency_ms.get("query_embedding"),
        "analytics": analytics_latency_ms,
        "retrieval": tracking_session.stage_latency_ms.get("retrieval"),
        "synthesis": synthesis_latency_ms,
        "validation": validation_latency_ms,
        "total": None,
    }
    query_run_id = _persist_answer_run(
        session,
        tracking_session,
        question=question,
        plan=plan,
        bundle=bundle,
        response_fields=response_fields,
        stage_latency_ms=stage_latency_ms,
        model_usage={
            "planner": _model_usage(planner_result.usage),
            "synthesizer": _model_usage(synthesizer_usage),
        },
        grounding_outcomes=grounding_outcomes,
        retrieval_outcomes=_retrieval_outcomes(bundle, synthesis),
        total_started=total_started,
    )
    return AnswerResponse(query_run_id=query_run_id, **response_fields)
