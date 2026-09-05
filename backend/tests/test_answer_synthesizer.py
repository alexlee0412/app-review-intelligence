from __future__ import annotations

import json
from datetime import datetime, timezone

from app.schemas.answer import (
    AppAggregate,
    EvidenceBundle,
    EvidenceItem,
    RetrievalTrace,
)
from app.schemas.query_plan import Intent, QueryPlan
from app.schemas.search import AppliedFilters, ReviewEvidence
from app.services.answer_synthesizer import synthesize_answer


class StubClient:
    name = "stub"
    is_production_grade = True

    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def complete_json(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.output


def _bundle() -> EvidenceBundle:
    body = "Stored review text with a tail that is not in the excerpt."
    evidence = [
        EvidenceItem(
            evidence_id=f"E{index}",
            review=ReviewEvidence(
                review_id=f"review-{index}",
                app_id="app-one",
                app_name="App One",
                version="1.0",
                rating=index,
                country="US",
                created_at=datetime(2026, 7, index, tzinfo=timezone.utc),
                title=None,
                body=body,
                similarity=0.9,
            ),
            excerpt="Stored review text",
        )
        for index in range(1, 4)
    ]
    plan = QueryPlan(
        intent=Intent.REVIEW_SUMMARY,
        semantic_query="subscription problems",
    )
    aggregate = AppAggregate(
        app_id="app-one",
        app_name="App One",
        review_count=12,
        matched_count=3,
        avg_rating=2.0,
    )
    return EvidenceBundle(
        query_plan=plan,
        applied_filters=AppliedFilters(countries=["US"]),
        aggregates=[aggregate],
        evidence=evidence,
        total_candidates=3,
        retrieval_trace=RetrievalTrace(
            semantic_query=plan.semantic_query,
            top_k=8,
            total_candidates=3,
            returned_evidence=3,
        ),
    )


def test_synthesizer_calls_once_with_only_allowed_evidence_text() -> None:
    client = StubClient(
        {
            "answer": "Users describe subscription problems [E1].",
            "findings": [
                {
                    "claim": "A stored review describes the issue.",
                    "evidence_ids": ["E1"],
                    "kind": "observed",
                }
            ],
        }
    )

    output, limitations = synthesize_answer(
        _bundle(), client=client, model="answer-model", max_output_tokens=500
    )

    assert len(client.calls) == 1
    assert output.findings[0].evidence_ids == ["E1"]
    assert limitations == []
    payload = json.loads(client.calls[0]["user"])
    assert payload["evidence"][0]["excerpt"] == "Stored review text"
    assert "tail that is not in the excerpt" not in client.calls[0]["user"]
    assert "review_id" not in payload["evidence"][0]
    assert payload["aggregates"][0]["matched_count"] == 3


def test_unknown_evidence_id_drops_finding_and_records_claim() -> None:
    claim = "An unsupported claim"
    client = StubClient(
        {
            "answer": "A response.",
            "findings": [
                {
                    "claim": claim,
                    "evidence_ids": ["E1", "E99"],
                    "kind": "interpretation",
                }
            ],
        }
    )

    output, limitations = synthesize_answer(
        _bundle(), client=client, model="answer-model", max_output_tokens=500
    )

    assert output.findings == []
    assert len(limitations) == 1
    assert claim in limitations[0]


def test_finding_without_citations_is_dropped() -> None:
    client = StubClient(
        {
            "answer": "A response.",
            "findings": [
                {"claim": "No support", "evidence_ids": [], "kind": "observed"}
            ],
        }
    )
    output, limitations = synthesize_answer(
        _bundle(), client=client, model="answer-model", max_output_tokens=500
    )
    assert output.findings == []
    assert limitations


def test_invalid_output_returns_safe_nonfabricated_result() -> None:
    client = StubClient({"answer": 123, "findings": "invalid"})
    output, limitations = synthesize_answer(
        _bundle(), client=client, model="answer-model", max_output_tokens=500
    )
    assert output.findings == []
    assert "could not produce a grounded answer" in output.answer
    assert limitations


def test_provider_response_error_returns_safe_result() -> None:
    class LLMResponseError(RuntimeError):
        pass

    class FailingClient(StubClient):
        def complete_json(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            raise LLMResponseError("provider response failed")

    client = FailingClient(None)
    output, limitations = synthesize_answer(
        _bundle(), client=client, model="answer-model", max_output_tokens=500
    )
    assert len(client.calls) == 1
    assert output.findings == []
    assert limitations


def test_programming_error_propagates() -> None:
    class FailingClient(StubClient):
        def complete_json(self, **kwargs: object) -> object:
            raise AttributeError("implementation error")

    try:
        synthesize_answer(
            _bundle(), client=FailingClient(None), model="answer-model", max_output_tokens=500
        )
    except AttributeError as exc:
        assert str(exc) == "implementation error"
    else:
        raise AssertionError("programming error did not propagate")
