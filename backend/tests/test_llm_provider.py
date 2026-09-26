"""Unit tests for structured completion providers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.services import llm_provider
from app.services.llm_provider import (
    FakeLLMClient,
    LLMConfigurationError,
    LLMResponseError,
    OpenAILLMClient,
    build_llm_client,
)

DATABASE_URL = "postgresql+psycopg://example:placeholder@db.example.invalid/app"
SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
}


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        app_environment="test",
        database_url=DATABASE_URL,
        **overrides,
    )


def test_fake_client_is_deterministic_and_not_production_grade() -> None:
    client = FakeLLMClient()
    arguments = {
        "model": "test-model",
        "system": "system",
        "user": "user",
        "schema": SIMPLE_SCHEMA,
        "max_output_tokens": 100,
    }
    baseline = client.complete_json(**arguments)
    assert baseline == client.complete_json(**arguments)
    assert baseline == client.complete_json(**arguments, reasoning_effort="high")
    assert client.name == "fake"
    assert client.is_production_grade is False


def test_fake_client_never_reports_usage() -> None:
    usage: list[dict[str, object]] = []
    FakeLLMClient().complete_json(
        model="test-model",
        system="system",
        user="user",
        schema=SIMPLE_SCHEMA,
        max_output_tokens=100,
        on_usage=usage.append,
    )
    assert usage == []


def test_builder_selects_fake() -> None:
    client = build_llm_client(_settings(llm_provider="fake"))
    assert isinstance(client, FakeLLMClient)


def test_openai_without_key_raises_without_exposing_configuration() -> None:
    settings = _settings(llm_provider="openai", openai_api_key=None)
    with pytest.raises(LLMConfigurationError) as caught:
        build_llm_client(settings)
    message = str(caught.value)
    assert "APP_OPENAI_API_KEY" in message
    assert DATABASE_URL not in message


class _Completions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _openai_client(completions: _Completions) -> OpenAILLMClient:
    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return OpenAILLMClient(SecretStr("placeholder-key"), client=transport)


def test_openai_requests_strict_json_and_retries_429(monkeypatch) -> None:
    completions = _Completions(
        [_StatusError(429), json.dumps({"value": "ok"})]
    )
    monkeypatch.setattr(llm_provider.time, "sleep", lambda _: None)
    client = _openai_client(completions)

    result = client.complete_json(
        model="test-model",
        system="system",
        user="user",
        schema=SIMPLE_SCHEMA,
        max_output_tokens=100,
    )

    assert result == {"value": "ok"}
    assert len(completions.calls) == 2
    response_format = completions.calls[-1]["response_format"]
    assert response_format["json_schema"]["strict"] is True
    strict_schema = response_format["json_schema"]["schema"]
    assert strict_schema["additionalProperties"] is False
    assert strict_schema["required"] == ["value"]
    assert "reasoning_effort" not in completions.calls[-1]


def test_openai_forwards_reasoning_effort_when_set() -> None:
    completions = _Completions([json.dumps({"value": "ok"})])
    client = _openai_client(completions)

    client.complete_json(
        model="test-model",
        system="system",
        user="user",
        schema=SIMPLE_SCHEMA,
        max_output_tokens=100,
        reasoning_effort="medium",
    )

    assert completions.calls[0]["reasoning_effort"] == "medium"


def test_openai_reports_normalized_provider_usage() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({"value": "ok"}))
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
        ),
    )

    class Completions:
        def create(self, **kwargs: object) -> object:
            return response

    usage: list[dict[str, object]] = []
    client = _openai_client(Completions())  # type: ignore[arg-type]
    client.complete_json(
        model="test-model",
        system="system",
        user="user",
        schema=SIMPLE_SCHEMA,
        max_output_tokens=100,
        on_usage=usage.append,
    )

    assert usage == [
        {
            "provider": "openai",
            "model": "test-model",
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "cached_input_tokens": 3,
        }
    ]


def test_openai_without_usage_does_not_invoke_callback() -> None:
    usage: list[dict[str, object]] = []
    client = _openai_client(_Completions([json.dumps({"value": "ok"})]))
    client.complete_json(
        model="test-model",
        system="system",
        user="user",
        schema=SIMPLE_SCHEMA,
        max_output_tokens=100,
        on_usage=usage.append,
    )
    assert usage == []


@pytest.mark.parametrize("content", ["not-json", "[]", "{}"])
def test_openai_rejects_invalid_or_schema_mismatched_output(content: str) -> None:
    client = _openai_client(_Completions([content]))
    with pytest.raises(LLMResponseError):
        client.complete_json(
            model="test-model",
            system="system",
            user="user",
            schema=SIMPLE_SCHEMA,
            max_output_tokens=100,
        )


def test_truncated_response_reports_the_token_budget() -> None:
    """A reasoning model can exhaust the budget before emitting any JSON."""

    class _Message:
        content = ""

    class _Choice:
        finish_reason = "length"
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **_: object) -> _Response:
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    provider = OpenAILLMClient(api_key=SecretStr("placeholder"), timeout_seconds=1)
    provider._client = _Client()

    with pytest.raises(LLMResponseError) as caught:
        provider.complete_json(
            model="gpt-5",
            system="s",
            user="u",
            schema={"type": "object", "properties": {}, "required": []},
            max_output_tokens=16,
        )

    assert "APP_LLM_MAX_OUTPUT_TOKENS" in str(caught.value)
