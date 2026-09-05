"""Grounded answer synthesis with citation validation."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.schemas.answer import EvidenceBundle, Finding, SynthesisOutput

_SYSTEM_PROMPT = """Return JSON matching the supplied schema.
Use [E#] citations for every factual claim.
Never invent apps, ratings, dates, counts, quotations, or evidence identifiers.
Quote evidence excerpts verbatim and state plainly when the evidence is insufficient.
Classify every finding as observed, computed, or interpretation.
Computed figures must come only from the supplied aggregates.
"""


def _payload(bundle: EvidenceBundle) -> dict[str, Any]:
    evidence = []
    for item in bundle.evidence:
        review = item.review
        evidence.append(
            {
                "evidence_id": item.evidence_id,
                "excerpt": item.excerpt,
                "app_id": review.app_id,
                "app_name": review.app_name,
                "version": review.version,
                "rating": review.rating,
                "country": review.country,
                "created_at": review.created_at.isoformat(),
                "title": review.title,
                "similarity": review.similarity,
            }
        )
    return {
        "intent": bundle.query_plan.intent.value,
        "semantic_query": bundle.query_plan.semantic_query,
        "evidence": evidence,
        "aggregates": [
            aggregate.model_dump(mode="json") for aggregate in bundle.aggregates
        ],
        "totals": bundle.totals,
    }


def _is_response_error(exc: Exception) -> bool:
    return any(cls.__name__ == "LLMResponseError" for cls in type(exc).__mro__)


def _safe_output(message: str) -> tuple[SynthesisOutput, list[str]]:
    return (
        SynthesisOutput(
            answer="I could not produce a grounded answer from the available evidence.",
            findings=[],
        ),
        [message],
    )


def _validate_findings(
    output: SynthesisOutput, evidence_ids: set[str]
) -> tuple[SynthesisOutput, list[str]]:
    findings: list[Finding] = []
    limitations: list[str] = []
    for finding in output.findings:
        unknown_ids = [item for item in finding.evidence_ids if item not in evidence_ids]
        if unknown_ids or not finding.evidence_ids:
            limitations.append(
                f"Dropped unsupported finding {finding.claim!r}; its citations did not "
                "resolve to retrieved evidence."
            )
            continue
        findings.append(finding)
    return output.model_copy(update={"findings": findings}), limitations


def synthesize_answer(
    bundle: EvidenceBundle,
    *,
    client: Any,
    model: str,
    max_output_tokens: int,
) -> tuple[SynthesisOutput, list[str]]:
    """Make one structured completion and discard unsupported findings."""
    try:
        raw_output = client.complete_json(
            model=model,
            system=_SYSTEM_PROMPT,
            user=json.dumps(_payload(bundle), ensure_ascii=False, sort_keys=True),
            schema=SynthesisOutput.model_json_schema(),
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        if not _is_response_error(exc):
            raise
        return _safe_output(
            "Answer synthesis was unavailable; no generated claims were returned."
        )

    try:
        output = SynthesisOutput.model_validate(raw_output)
    except (TypeError, ValidationError):
        return _safe_output(
            "Answer synthesis returned an invalid response; no generated claims were "
            "returned."
        )
    return _validate_findings(output, bundle.evidence_ids)
