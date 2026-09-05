"""Validated question planning constrained to the available review dataset."""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.query_plan import (
    DEFAULT_PLAN_TOP_K,
    AppCatalogEntry,
    Intent,
    PlannerResult,
    QueryPlan,
)
from app.services.llm_provider import LLMClient, LLMResponseError

_SYSTEM_PROMPT = """Interpret one question as a strict QueryPlan JSON object.
The fixed dataset scope is the US storefront and iOS platform. A plan may narrow this
scope but must never widen it. Resolve applications only to identifiers in the supplied
catalog; never invent an app identifier. Do not compute results or produce SQL.
"""

_FALLBACK_LIMITATION = (
    "Question planner returned invalid output twice; used deterministic semantic "
    "fallback."
)


def _planner_input(question: str, catalog: list[AppCatalogEntry]) -> str:
    return json.dumps(
        {
            "question": question,
            "scope": {"countries": ["US"], "platform": "ios"},
            "catalog": [entry.model_dump(mode="json") for entry in catalog],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _fallback_plan(question: str) -> QueryPlan:
    return QueryPlan(
        intent=Intent.SEMANTIC_EVIDENCE,
        semantic_query=question,
        top_k=DEFAULT_PLAN_TOP_K,
    )


def plan_question(
    question: str,
    catalog: list[AppCatalogEntry],
    *,
    client: LLMClient,
    settings: Settings,
) -> PlannerResult:
    """Plan one question, retrying invalid output once before a safe fallback."""
    user_prompt = _planner_input(question, catalog)
    plan: QueryPlan | None = None

    for _ in range(2):
        try:
            output = client.complete_json(
                model=settings.planner_model,
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                schema=QueryPlan.model_json_schema(),
                max_output_tokens=settings.llm_max_output_tokens,
            )
            plan = QueryPlan.model_validate(output)
            break
        except (LLMResponseError, ValidationError, TypeError, ValueError):
            continue

    used_fallback = plan is None
    limitations = [_FALLBACK_LIMITATION] if used_fallback else []
    if plan is None:
        plan = _fallback_plan(question)

    plan, scope_limitations = plan.restrict_apps_to(
        {entry.app_id for entry in catalog}
    )
    limitations.extend(scope_limitations)
    return PlannerResult(
        plan=plan,
        model=settings.planner_model,
        used_fallback=used_fallback,
        limitations=limitations,
    )
