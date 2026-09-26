"""Unit tests for validated and scope-restricted question planning."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.config import Settings
from app.schemas.query_plan import (
    DEFAULT_PLAN_TOP_K,
    AppCatalogEntry,
    Intent,
)
from app.services.llm_provider import LLMResponseError
from app.services.query_planner import plan_question

DATABASE_URL = "postgresql+psycopg://example:placeholder@db.example.invalid/app"
CATALOG = [
    AppCatalogEntry(app_id="app-one", app_name="App One"),
    AppCatalogEntry(app_id="app-two", app_name="App Two"),
]


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        app_environment="test",
        database_url=DATABASE_URL,
        llm_provider="fake",
    )


class ScriptedClient:
    name = "fake"
    is_production_grade = False

    def __init__(self, outcomes: list[dict[str, Any] | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs) -> dict[str, Any]:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _valid_plan(**overrides) -> dict[str, Any]:
    return {
        "intent": "semantic_evidence",
        "semantic_query": "subscription cancellation",
        **overrides,
    }


def test_valid_plan_is_parsed_with_catalog_in_prompt() -> None:
    client = ScriptedClient([_valid_plan(app_ids=["app-one"])])
    result = plan_question(
        "Why is cancellation hard?",
        CATALOG,
        client=client,
        settings=_settings(),
    )
    prompt = json.loads(client.calls[0]["user"])

    assert result.plan.intent is Intent.SEMANTIC_EVIDENCE
    assert result.plan.app_ids == ["app-one"]
    assert result.used_fallback is False
    assert len(client.calls) == 1
    assert {item["app_id"] for item in prompt["catalog"]} == {
        "app-one",
        "app-two",
    }
    assert prompt["scope"] == {"countries": ["US"], "platform": "ios"}


def test_planner_carries_exact_usage_from_provider() -> None:
    class UsageClient(ScriptedClient):
        def complete_json(self, **kwargs) -> dict[str, Any]:
            kwargs["on_usage"](
                {
                    "provider": "openai",
                    "model": "planner-model",
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                    "cached_input_tokens": None,
                }
            )
            return super().complete_json(**kwargs)

    result = plan_question(
        "Why is cancellation hard?",
        CATALOG,
        client=UsageClient([_valid_plan()]),
        settings=_settings(),
    )

    assert result.usage == {
        "provider": "openai",
        "model": "planner-model",
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
        "cached_input_tokens": None,
    }


def test_malformed_output_retries_once_then_falls_back() -> None:
    client = ScriptedClient(
        [
            LLMResponseError("malformed"),
            LLMResponseError("malformed again"),
        ]
    )
    question = "What do reviewers dislike?"
    result = plan_question(
        question,
        CATALOG,
        client=client,
        settings=_settings(),
    )

    assert len(client.calls) == 2
    assert result.used_fallback is True
    assert result.plan.intent is Intent.SEMANTIC_EVIDENCE
    assert result.plan.semantic_query == question
    assert result.plan.top_k == DEFAULT_PLAN_TOP_K
    assert any("fallback" in limitation for limitation in result.limitations)


def test_retry_can_recover_with_valid_plan() -> None:
    client = ScriptedClient(
        [LLMResponseError("malformed"), _valid_plan(app_ids=["app-two"])]
    )
    result = plan_question(
        "Question",
        CATALOG,
        client=client,
        settings=_settings(),
    )
    assert len(client.calls) == 2
    assert result.used_fallback is False
    assert result.plan.app_ids == ["app-two"]


def test_unknown_app_id_is_dropped_with_limitation() -> None:
    client = ScriptedClient(
        [_valid_plan(app_ids=["app-one", "invented-app"])]
    )
    result = plan_question(
        "Compare apps",
        CATALOG,
        client=client,
        settings=_settings(),
    )
    assert result.plan.app_ids == ["app-one"]
    assert result.dropped_app_ids == ["invented-app"]
    assert any("invented-app" in limitation for limitation in result.limitations)


@pytest.mark.parametrize("intent", ["unsupported", "trend_analysis"])
def test_non_answering_intents_disable_semantic_search(intent: str) -> None:
    client = ScriptedClient(
        [
            {
                "intent": intent,
                "semantic_query": None,
                "needs_semantic_search": True,
                "needs_aggregation": True,
            }
        ]
    )
    result = plan_question(
        "Question",
        CATALOG,
        client=client,
        settings=_settings(),
    )
    assert result.plan.needs_semantic_search is False
    assert result.plan.needs_aggregation is False


def test_country_is_clamped_to_supported_scope() -> None:
    client = ScriptedClient([_valid_plan(countries=["KR"])])
    result = plan_question(
        "Question",
        CATALOG,
        client=client,
        settings=_settings(),
    )
    assert result.plan.countries == ["US"]


def test_out_of_bounds_top_k_uses_bounded_fallback() -> None:
    client = ScriptedClient(
        [_valid_plan(top_k=10_000), _valid_plan(top_k=10_000)]
    )
    result = plan_question(
        "Question",
        CATALOG,
        client=client,
        settings=_settings(),
    )
    assert result.used_fallback is True
    assert result.plan.top_k == DEFAULT_PLAN_TOP_K
