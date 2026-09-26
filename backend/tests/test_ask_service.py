from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.db import engine
from app.models import EMBEDDING_DIMENSION, App, QueryRun, Review
from app.models.query_run import RUN_KIND_ASK
from app.repositories.review_search_repository import SearchQueryResult
from app.schemas.answer import (
    AppAggregate,
    EvidenceBundle,
    EvidenceItem,
    RetrievalTrace,
)
from app.schemas.query_plan import Intent, PlannerResult, QueryPlan
from app.schemas.search import AppliedFilters, QueryEmbedder, ReviewEvidence
from app.services import ask_service
from app.services import review_search
from app.services.ask_service import AskOverrides, answer_question
from app.services.embedding_service import EmbeddingConfigurationError


class FakeResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class FakeSession:
    def __init__(self) -> None:
        self.catalog = [SimpleNamespace(app_id="app-one", app_name="App One")]
        self.added: list[object] = []
        self.commits = 0

    def execute(self, statement: object) -> FakeResult:
        return FakeResult(self.catalog)

    def add(self, value: object) -> None:
        self.added.append(value)

    def commit(self) -> None:
        self.commits += 1


class StubClient:
    def __init__(self, *, production: bool = True) -> None:
        self.name = "openai" if production else "fake"
        self.is_production_grade = production
        self.calls: list[dict[str, object]] = []

    def complete_json(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "answer": "Customers describe cancellation problems [E1].",
            "findings": [
                {
                    "claim": "A customer describes cancellation problems.",
                    "evidence_ids": ["E1"],
                    "kind": "observed",
                }
            ],
            "metrics": {"matched_count": 9999},
        }


def _settings(*, reasoning_effort: str | None = None) -> object:
    return SimpleNamespace(
        planner_model="planner-model",
        synthesizer_model="answer-model",
        synthesizer_reasoning_effort=reasoning_effort,
        llm_max_output_tokens=500,
    )


def _embedder(*, production: bool = True) -> QueryEmbedder:
    return QueryEmbedder(
        name="openai" if production else "fake",
        is_production_grade=production,
        embed=lambda _: [0.0] * EMBEDDING_DIMENSION,
    )


def _planner(plan: QueryPlan):
    def plan_question(*args: object, **kwargs: object) -> PlannerResult:
        return PlannerResult(plan=plan, model="planner-model")

    return plan_question


def _bundle(
    plan: QueryPlan,
    count: int,
    *,
    embedding_production: bool = True,
) -> EvidenceBundle:
    evidence = [
        EvidenceItem(
            evidence_id=f"E{index}",
            review=ReviewEvidence(
                review_id=f"review-{index}",
                app_id="app-one",
                app_name="App One",
                version="1.0",
                rating=1,
                country="US",
                created_at=datetime(2026, 7, index, tzinfo=timezone.utc),
                title=None,
                body=f"Stored review {index}",
                similarity=0.9,
            ),
            excerpt=f"Stored review {index}",
        )
        for index in range(1, count + 1)
    ]
    return EvidenceBundle(
        query_plan=plan,
        applied_filters=AppliedFilters(
            app_ids=plan.app_ids,
            countries=plan.countries,
            ratings=plan.ratings,
        ),
        evidence=evidence,
        total_candidates=count,
        limitations=[] if count >= 3 else ["Evidence is too thin."],
        retrieval_trace=RetrievalTrace(
            semantic_query=plan.semantic_query,
            top_k=plan.top_k,
            total_candidates=count,
            returned_evidence=count,
            embedding_provider="openai" if embedding_production else "fake",
            embedding_is_production_grade=embedding_production,
            similarity_metric="cosine",
            sql_template="SELECT parameterized",
        ),
    )


def _install_bundle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
    embedding_production: bool = True,
) -> None:
    monkeypatch.setattr(
        ask_service,
        "build_evidence",
        lambda session, plan, **kwargs: _bundle(
            plan, count, embedding_production=embedding_production
        ),
    )


def test_thin_evidence_skips_synthesis_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = QueryPlan(
        intent=Intent.SEMANTIC_EVIDENCE,
        semantic_query="cancellation",
        app_ids=["app-one"],
    )
    _install_bundle(monkeypatch, count=2)
    client = StubClient()

    response = answer_question(
        FakeSession(),  # type: ignore[arg-type]
        "What are customers saying?",
        planner=_planner(plan),
        embedder=_embedder(),
        client=client,
        settings=_settings(),  # type: ignore[arg-type]
    )

    assert client.calls == []
    assert response.trace.synthesis_skipped is True
    assert "Insufficient evidence" in response.answer


@pytest.mark.parametrize("intent", [Intent.UNSUPPORTED, Intent.TREND_ANALYSIS])
def test_nonanswering_intents_skip_synthesis(
    monkeypatch: pytest.MonkeyPatch, intent: Intent
) -> None:
    plan = QueryPlan(intent=intent, semantic_query=None)
    _install_bundle(monkeypatch, count=0)
    client = StubClient()

    response = answer_question(
        FakeSession(),  # type: ignore[arg-type]
        "Question outside supported scope",
        planner=_planner(plan),
        embedder=_embedder(),
        client=client,
        settings=_settings(),  # type: ignore[arg-type]
    )

    assert client.calls == []
    assert response.trace.synthesis_skipped is True
    assert response.findings == []


def test_metrics_are_copied_from_aggregates_not_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = QueryPlan(
        intent=Intent.APP_COMPARISON,
        semantic_query="cancellation",
        app_ids=["app-one"],
        requested_metrics=["matched_count", "avg_rating"],
    )
    _install_bundle(monkeypatch, count=3)
    aggregate = AppAggregate(
        app_id="app-one",
        app_name="App One",
        review_count=50,
        matched_count=3,
        avg_rating=1.5,
        rating_distribution={1: 2, 2: 1},
    )
    session = FakeSession()

    response = answer_question(
        session,  # type: ignore[arg-type]
        "Compare cancellation reviews",
        planner=_planner(plan),
        aggregate_builder=lambda *_: [aggregate],
        totals_builder=lambda items: {"total_matched": items[0].matched_count},
        embedder=_embedder(),
        client=StubClient(),
        settings=_settings(),  # type: ignore[arg-type]
    )

    assert response.metrics == {
        "totals": {"total_matched": 3},
        "apps": [aggregate.model_dump(mode="json")],
    }
    assert response.metrics["apps"][0]["matched_count"] == 3
    assert len(session.added) == 1
    query_run = session.added[0]
    assert isinstance(query_run, QueryRun)
    assert query_run.user_query == "Compare cancellation reviews"
    assert query_run.parsed_intent["intent"] == "app_comparison"
    assert query_run.sql_template == "SELECT parameterized"
    assert session.commits == 1


def test_reasoning_effort_is_forwarded_to_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = QueryPlan(
        intent=Intent.REVIEW_SUMMARY,
        semantic_query="cancellation",
        app_ids=["app-one"],
    )
    _install_bundle(monkeypatch, count=3)
    client = StubClient()

    answer_question(
        FakeSession(),  # type: ignore[arg-type]
        "Summarize cancellation reviews",
        planner=_planner(plan),
        embedder=_embedder(),
        client=client,
        settings=_settings(reasoning_effort="high"),  # type: ignore[arg-type]
    )

    assert client.calls[0]["reasoning_effort"] == "high"


def test_nonproduction_providers_surface_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = QueryPlan(
        intent=Intent.REVIEW_SUMMARY,
        semantic_query="cancellation",
        app_ids=["app-one"],
    )
    _install_bundle(monkeypatch, count=3, embedding_production=False)

    response = answer_question(
        FakeSession(),  # type: ignore[arg-type]
        "Summarize cancellation reviews",
        planner=_planner(plan),
        embedder=_embedder(production=False),
        client=StubClient(production=False),
        settings=_settings(),  # type: ignore[arg-type]
    )

    assert len(response.warnings) == 2
    assert any("non-production answer provider" in item for item in response.warnings)
    assert any("non-semantic development embedding" in item for item in response.warnings)


def test_app_override_accepts_name_and_drops_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = QueryPlan(intent=Intent.REVIEW_SUMMARY, semantic_query="cancellation")
    captured: list[QueryPlan] = []

    def bundle_for_override(session: object, used_plan: QueryPlan, **kwargs: object):
        captured.append(used_plan)
        return _bundle(used_plan, 2)

    monkeypatch.setattr(ask_service, "build_evidence", bundle_for_override)
    response = answer_question(
        FakeSession(),  # type: ignore[arg-type]
        "Summarize cancellation reviews",
        overrides=AskOverrides(apps=["App One", "missing"], ratings=[1, 2], top_k=4),
        planner=_planner(plan),
        embedder=_embedder(),
        client=StubClient(),
        settings=_settings(),  # type: ignore[arg-type]
    )

    assert captured[0].app_ids == ["app-one"]
    assert captured[0].ratings == [1, 2]
    assert captured[0].top_k == 4
    assert any("missing" in limitation for limitation in response.limitations)


def test_planner_that_dropped_all_apps_cannot_widen_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = QueryPlan(
        intent=Intent.REVIEW_SUMMARY,
        semantic_query="cancellation",
        app_ids=["invented"],
    )
    plan, limitations, dropped_app_ids = planned.restrict_apps_to({"app-one"})
    assert limitations == [
        "Ignored unknown app identifier 'invented'; it is not in the dataset."
    ]
    planner_result = PlannerResult(
        plan=plan,
        model="planner-model",
        limitations=["The nonexistent app 'invented' was removed from scope."],
        dropped_app_ids=dropped_app_ids,
    )
    captured: list[QueryPlan] = []

    def planner(*args: object, **kwargs: object) -> PlannerResult:
        return planner_result

    def capture_bundle(session: object, used_plan: QueryPlan, **kwargs: object):
        captured.append(used_plan)
        return _bundle(used_plan, 0)

    monkeypatch.setattr(ask_service, "build_evidence", capture_bundle)
    response = answer_question(
        FakeSession(),  # type: ignore[arg-type]
        "What does the invented app say?",
        planner=planner,
        embedder=_embedder(),
        client=StubClient(),
        settings=_settings(),  # type: ignore[arg-type]
    )

    assert captured[0].needs_semantic_search is False
    assert response.trace.synthesis_skipped is True
    assert "invented" in response.limitations[0]


def _install_search_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        review_search,
        "search_review_candidates",
        lambda **kwargs: SearchQueryResult(
            evidence=[
                ReviewEvidence(
                    review_id=f"review-{index}",
                    app_id="app-one",
                    app_name="App One",
                    version="1.0",
                    rating=1,
                    country="US",
                    created_at=datetime(2026, 7, index, tzinfo=timezone.utc),
                    title=None,
                    body=f"Stored review {index}",
                    similarity=0.9,
                )
                for index in range(1, 4)
            ],
            matched_review_count=3,
            sql_template="SELECT parameterized",
        ),
    )


def test_ask_retrieval_row_is_ask_before_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_search_candidates(monkeypatch)
    session = FakeSession()
    plan = QueryPlan(
        intent=Intent.REVIEW_SUMMARY,
        semantic_query="cancellation",
        app_ids=["app-one"],
        top_k=3,
    )

    answer_question(
        session,  # type: ignore[arg-type]
        "Summarize cancellation reviews",
        planner=_planner(plan),
        embedder=_embedder(),
        client=StubClient(),
        settings=_settings(),  # type: ignore[arg-type]
    )

    query_runs = [item for item in session.added if isinstance(item, QueryRun)]
    assert len({item.query_run_id for item in query_runs}) == 1
    assert all(item.run_kind == RUN_KIND_ASK for item in query_runs)
    assert session.commits == 2


def test_failed_ask_after_retrieval_is_never_labeled_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_search_candidates(monkeypatch)
    session = FakeSession()
    plan = QueryPlan(
        intent=Intent.REVIEW_SUMMARY,
        semantic_query="cancellation",
        app_ids=["app-one"],
        top_k=3,
    )

    class FailingClient(StubClient):
        def complete_json(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("synthesis failed")

    with pytest.raises(RuntimeError, match="synthesis failed"):
        answer_question(
            session,  # type: ignore[arg-type]
            "Summarize cancellation reviews",
            planner=_planner(plan),
            embedder=_embedder(),
            client=FailingClient(),
            settings=_settings(),  # type: ignore[arg-type]
        )

    query_runs = [item for item in session.added if isinstance(item, QueryRun)]
    assert len(query_runs) == 1
    assert query_runs[0].run_kind == RUN_KIND_ASK
    assert session.commits == 1


def test_default_embedder_configuration_error_remains_a_domain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = QueryPlan(
        intent=Intent.REVIEW_SUMMARY,
        semantic_query="cancellation",
        app_ids=["app-one"],
    )

    def fail_build(settings: object) -> QueryEmbedder:
        raise EmbeddingConfigurationError("embedding configuration is unavailable")

    monkeypatch.setattr(ask_service, "_build_embedder", fail_build)

    with pytest.raises(
        EmbeddingConfigurationError, match="embedding configuration is unavailable"
    ):
        answer_question(
            FakeSession(),  # type: ignore[arg-type]
            "Summarize cancellation reviews",
            planner=_planner(plan),
            client=StubClient(),
            settings=_settings(),  # type: ignore[arg-type]
        )


def test_missing_dependencies_are_imported_only_inside_ask_service_functions() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    ask_source = (backend_root / "app/services/ask_service.py").read_text()
    provider_import = "from app.services." + "llm_provider import build_llm_client"
    planner_import = "from app.services." + "query_planner import plan_question"

    assert ask_source.index(provider_import) > ask_source.index("def _build_client")
    assert ask_source.index(planner_import) > ask_source.index("def _default_planner")
    for relative_path in (
        "app/services/evidence_builder.py",
        "app/services/answer_synthesizer.py",
    ):
        source = (backend_root / relative_path).read_text()
        assert "app.services." + "llm_provider" not in source
        assert "app.services." + "query_planner" not in source


def test_service_modules_do_not_import_api_modules() -> None:
    services = Path(__file__).resolve().parents[1] / "app/services"
    for path in services.glob("*.py"):
        assert "from app.api" not in path.read_text(), path.name
        assert "import app.api" not in path.read_text(), path.name


@pytest.fixture
def integration_session() -> Generator[Session, None, None]:
    try:
        connection = engine.connect()
    except SQLAlchemyError:
        pytest.skip("PostgreSQL is not available")
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.mark.integration
def test_answer_question_uses_only_owned_rows_and_persists_one_trace(
    integration_session: Session,
) -> None:
    suffix = uuid.uuid4().hex
    app_id = f"ask-{suffix}"
    review_ids = [f"ask-review-{suffix}-{index}" for index in range(3)]
    integration_session.add(App(app_id=app_id, app_name="Owned Ask App"))
    integration_session.flush()
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = 1.0
    integration_session.add_all(
        [
            Review(
                review_id=review_id,
                app_id=app_id,
                rating=1,
                country="US",
                created_at=datetime(2026, 7, index + 1, tzinfo=timezone.utc),
                body=f"Owned evidence body {index}",
                embedding=vector,
            )
            for index, review_id in enumerate(review_ids)
        ]
    )
    integration_session.flush()
    before = integration_session.scalar(select(func.count()).select_from(QueryRun)) or 0
    plan = QueryPlan(
        intent=Intent.SEMANTIC_EVIDENCE,
        semantic_query="owned evidence",
        app_ids=[app_id],
        top_k=3,
    )

    response = answer_question(
        integration_session,
        "What does the owned evidence say?",
        planner=_planner(plan),
        embedder=QueryEmbedder(
            name="test",
            is_production_grade=True,
            embed=lambda _: vector,
        ),
        client=StubClient(),
        settings=_settings(),  # type: ignore[arg-type]
    )

    after = integration_session.scalar(select(func.count()).select_from(QueryRun)) or 0
    assert after == before + 1
    assert {item.review.review_id for item in response.evidence} == set(review_ids)
    query_run = integration_session.get(QueryRun, response.query_run_id)
    assert query_run is not None
    assert query_run.user_query == "What does the owned evidence say?"
    assert query_run.parsed_intent["intent"] == "semantic_evidence"
    assert query_run.run_kind == RUN_KIND_ASK
    assert query_run.stage_latency_ms["planner"] is not None
    assert query_run.stage_latency_ms["query_embedding"] is not None
    assert query_run.stage_latency_ms["analytics"] is None
    assert query_run.stage_latency_ms["retrieval"] is not None
    assert query_run.stage_latency_ms["synthesis"] is not None
    assert query_run.stage_latency_ms["validation"] is not None
    assert query_run.stage_latency_ms["total"] is not None
