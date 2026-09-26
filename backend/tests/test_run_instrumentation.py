from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.models import EMBEDDING_DIMENSION, QueryRun
from app.models.query_run import RUN_KIND_ASK, RUN_KIND_SEARCH, SCHEMA_VERSION
from app.schemas.answer import EvidenceBundle, EvidenceItem, RetrievalTrace
from app.schemas.query_plan import Intent, PlannerResult, QueryPlan
from app.schemas.search import AppliedFilters, QueryEmbedder, ReviewEvidence
from app.services import ask_service
from app.services.answer_synthesizer import synthesize_answer
from app.services.ask_service import answer_question


class FakeResult:
    def all(self) -> list[object]:
        return [SimpleNamespace(app_id="app-one", app_name="App One")]


class FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    def execute(self, statement: object) -> FakeResult:
        return FakeResult()

    def add(self, value: object) -> None:
        self.added.append(value)

    def commit(self) -> None:
        pass


class UsageClient:
    name = "openai"
    is_production_grade = True

    def complete_json(self, **kwargs: object) -> dict[str, object]:
        callback = kwargs["on_usage"]
        assert callable(callback)
        callback(
            {
                "provider": "openai",
                "model": kwargs["model"],
                "input_tokens": 30,
                "output_tokens": 10,
                "total_tokens": 40,
                "cached_input_tokens": 4,
            }
        )
        return {
            "answer": "Customers describe cancellation problems [E1].",
            "findings": [
                {
                    "claim": "A customer describes cancellation problems.",
                    "evidence_ids": ["E1"],
                    "kind": "observed",
                }
            ],
        }


def _plan() -> QueryPlan:
    return QueryPlan(
        intent=Intent.SEMANTIC_EVIDENCE,
        semantic_query="cancellation",
        app_ids=["app-one"],
        top_k=3,
    )


def _bundle(plan: QueryPlan, count: int = 3) -> EvidenceBundle:
    evidence = [
        EvidenceItem(
            evidence_id=f"E{position}",
            review=ReviewEvidence(
                review_id=f"review-{position}",
                app_id="app-one",
                app_name="App One",
                version="1.0",
                rating=1,
                country="US",
                created_at=datetime(2026, 7, position, tzinfo=timezone.utc),
                title=None,
                body=f"Stored review {position}",
                similarity=0.9,
            ),
            excerpt=f"Stored review {position}",
        )
        for position in range(1, count + 1)
    ]
    return EvidenceBundle(
        query_plan=plan,
        applied_filters=AppliedFilters(app_ids=plan.app_ids, countries=["US"]),
        evidence=evidence,
        total_candidates=count,
        totals={"total_reviews": 10},
        retrieval_trace=RetrievalTrace(
            semantic_query=plan.semantic_query,
            top_k=plan.top_k,
            total_candidates=count,
            returned_evidence=count,
            embedding_provider="openai",
            embedding_is_production_grade=True,
            similarity_metric="cosine",
            sql_template="SELECT parameterized",
        ),
    )


def _settings() -> object:
    return SimpleNamespace(
        planner_model="planner-model",
        synthesizer_model="answer-model",
        synthesizer_reasoning_effort=None,
        llm_max_output_tokens=500,
    )


def _embedder() -> QueryEmbedder:
    return QueryEmbedder(
        name="openai",
        is_production_grade=True,
        embed=lambda _: [0.0] * EMBEDDING_DIMENSION,
    )


def _contains_vector(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_vector(item) for item in value.values())
    if isinstance(value, list):
        return len(value) == EMBEDDING_DIMENSION or any(
            _contains_vector(item) for item in value
        )
    return False


def test_synthesis_reports_grounding_decisions_without_changing_them() -> None:
    bundle = _bundle(_plan())

    class GroundingClient:
        def complete_json(self, **kwargs: object) -> dict[str, object]:
            return {
                "answer": "A cancellation issue was reported [E1].",
                "findings": [
                    {
                        "claim": "A cancellation issue was reported.",
                        "evidence_ids": ["E1"],
                        "kind": "observed",
                    },
                    {
                        "claim": "An unsupported citation was supplied.",
                        "evidence_ids": ["E99"],
                        "kind": "observed",
                    },
                    {
                        "claim": "There were 999 complaints.",
                        "evidence_ids": ["E1"],
                        "kind": "computed",
                    },
                ],
            }

    result = synthesize_answer(
        bundle,
        client=GroundingClient(),
        model="answer-model",
        max_output_tokens=500,
    )

    assert [finding.claim for finding in result.output.findings] == [
        "A cancellation issue was reported."
    ]
    assert len(result.limitations) == 2
    assert result.grounding_outcomes == {
        "findings_before": 3,
        "findings_kept": 1,
        "findings_dropped": 2,
        "dropped_by_reason": {
            "unresolved_citation": 1,
            "unsupported_number": 1,
        },
        "kept_by_kind": {
            "observed": 1,
            "computed": 0,
            "interpretation": 0,
        },
        "limitations_emitted": 2,
        "narrative_withheld": False,
    }


def test_synthesis_reports_narrative_withheld() -> None:
    class NarrativeClient:
        def complete_json(self, **kwargs: object) -> dict[str, object]:
            return {"answer": "There were 999 complaints [E1].", "findings": []}

    result = synthesize_answer(
        _bundle(_plan()),
        client=NarrativeClient(),
        model="answer-model",
        max_output_tokens=500,
    )

    assert result.grounding_outcomes is not None
    assert result.grounding_outcomes["narrative_withheld"] is True
    assert result.grounding_outcomes["limitations_emitted"] == 1


def test_complete_ask_overwrites_search_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    planner_usage = {
        "provider": "openai",
        "model": "planner-model",
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
        "cached_input_tokens": None,
    }

    def planner(*args: object, **kwargs: object) -> PlannerResult:
        return PlannerResult(plan=plan, model="planner-model", usage=planner_usage)

    def build_evidence(session: object, used_plan: QueryPlan, **kwargs: object):
        query_run = QueryRun(query_run_id=uuid.uuid4(), run_kind=RUN_KIND_SEARCH)
        session.add(query_run)
        session.record_stage_latency("query_embedding", 1.25)
        session.record_stage_latency("retrieval", 2.5)
        return _bundle(used_plan)

    monkeypatch.setattr(ask_service, "build_evidence", build_evidence)
    session = FakeSession()
    answer_question(
        session,  # type: ignore[arg-type]
        "What do reviewers say?",
        planner=planner,
        embedder=_embedder(),
        client=UsageClient(),
        settings=_settings(),  # type: ignore[arg-type]
    )

    query_run = session.added[-1]
    assert isinstance(query_run, QueryRun)
    assert query_run is session.added[0]
    assert query_run.run_kind == RUN_KIND_ASK
    assert query_run.stage_latency_ms.keys() == {
        "planner",
        "query_embedding",
        "analytics",
        "retrieval",
        "synthesis",
        "validation",
        "total",
    }
    assert query_run.stage_latency_ms["planner"] >= 0
    assert query_run.stage_latency_ms["query_embedding"] == 1.25
    assert query_run.stage_latency_ms["analytics"] is None
    assert query_run.stage_latency_ms["retrieval"] == 2.5
    assert query_run.stage_latency_ms["synthesis"] >= 0
    assert query_run.stage_latency_ms["validation"] >= 0
    assert query_run.stage_latency_ms["total"] >= 0
    assert query_run.model_usage == {
        "planner": {**planner_usage, "cost": None},
        "synthesizer": {
            "provider": "openai",
            "model": "answer-model",
            "input_tokens": 30,
            "output_tokens": 10,
            "total_tokens": 40,
            "cached_input_tokens": 4,
            "cost": None,
        },
    }
    assert query_run.retrieval_outcomes == {
        "returned_evidence_ids": ["E1", "E2", "E3"],
        "cited_evidence_ids": ["E1"],
        "returned_count": 3,
        "cited_count": 1,
        "cited_ratio": pytest.approx(1 / 3),
    }
    assert query_run.run_versions == {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": {
            "planner": "planner-v1",
            "synthesizer": "synthesizer-v1",
        },
        "taxonomy_version": None,
    }
    instrumentation = {
        "stage_latency_ms": query_run.stage_latency_ms,
        "model_usage": query_run.model_usage,
        "grounding_outcomes": query_run.grounding_outcomes,
        "retrieval_outcomes": query_run.retrieval_outcomes,
        "run_versions": query_run.run_versions,
    }
    assert "Return JSON matching" not in repr(instrumentation)
    assert "Stored review" not in repr(instrumentation)
    assert _contains_vector(instrumentation) is False


def test_skipped_stages_and_empty_retrieval_are_explicitly_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setattr(
        ask_service,
        "build_evidence",
        lambda session, used_plan, **kwargs: _bundle(used_plan, count=0),
    )
    session = FakeSession()
    answer_question(
        session,  # type: ignore[arg-type]
        "What do reviewers say?",
        planner=lambda *args, **kwargs: PlannerResult(
            plan=plan, model="planner-model"
        ),
        embedder=_embedder(),
        client=UsageClient(),
        settings=_settings(),  # type: ignore[arg-type]
    )

    query_run = session.added[-1]
    assert isinstance(query_run, QueryRun)
    assert query_run.stage_latency_ms["query_embedding"] is None
    assert query_run.stage_latency_ms["analytics"] is None
    assert query_run.stage_latency_ms["retrieval"] is None
    assert query_run.stage_latency_ms["synthesis"] is None
    assert query_run.stage_latency_ms["validation"] is None
    assert query_run.model_usage == {"planner": None, "synthesizer": None}
    assert query_run.grounding_outcomes is None
    assert query_run.retrieval_outcomes == {
        "returned_evidence_ids": [],
        "cited_evidence_ids": [],
        "returned_count": 0,
        "cited_count": 0,
        "cited_ratio": None,
    }
