"""Unit tests for deterministic and production embedding providers."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.models.review import EMBEDDING_DIMENSION
from app.services.embedding_service import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedding_input,
    build_embedding_provider,
)

REMOTE_DATABASE_URL = "postgresql+psycopg://example:placeholder@db.example.invalid/app"


def settings(**overrides) -> Settings:
    values = {
        "database_url": REMOTE_DATABASE_URL,
        "app_environment": "test",
        "app_host": "example.invalid",
        "app_port": 9999,
        **overrides,
    }
    return Settings(_env_file=None, **values)


def test_fake_is_deterministic_normalized_and_expected_dimension() -> None:
    provider = FakeEmbeddingProvider()
    first = provider.embed_texts(["same review"])[0]
    second = provider.embed_texts(["same review"])[0]
    assert first == second
    assert len(first) == EMBEDDING_DIMENSION == 1536
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)
    assert provider.name == "fake"
    assert provider.is_production_grade is False


def test_build_embedding_input_with_and_without_title() -> None:
    assert build_embedding_input("Title", "Body") == "Title\n\nBody"
    assert build_embedding_input(None, "Body") == "Body"


def test_builder_selects_fake() -> None:
    provider = build_embedding_provider(settings(embedding_provider="fake"))
    assert isinstance(provider, FakeEmbeddingProvider)


def test_builder_selects_openai_and_reports_production_grade() -> None:
    provider = build_embedding_provider(
        settings(
            embedding_provider="openai",
            openai_api_key=SecretStr("placeholder-key"),
        )
    )
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.name == "openai"
    assert provider.is_production_grade is True


def test_openai_without_key_raises_clear_typed_error() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="APP_OPENAI_API_KEY"):
        build_embedding_provider(settings(embedding_provider="openai"))


def test_openai_dimension_mismatch_raises() -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
    )
    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **_: response)
    )
    provider = OpenAIEmbeddingProvider(
        SecretStr("placeholder-key"),
        client=client,
    )
    with pytest.raises(EmbeddingDimensionError):
        provider.embed_texts(["review"])
