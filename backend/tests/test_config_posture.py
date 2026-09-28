"""Configuration tests for credential redaction and deployment posture."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.models.review import EMBEDDING_DIMENSION

REMOTE_DATABASE_URL = "postgresql+psycopg://example:placeholder@db.example.invalid/app"
BASE_SETTINGS = {
    "app_host": "example.invalid",
    "app_port": 9999,
}


def test_embedding_dimension_agrees_with_model() -> None:
    settings = Settings(
        _env_file=None, database_url=REMOTE_DATABASE_URL, **BASE_SETTINGS
    )
    assert settings.embedding_dimension == EMBEDDING_DIMENSION


def test_embedding_dimension_mismatch_raises() -> None:
    with pytest.raises(ValidationError, match="must be 1536"):
        Settings(
            _env_file=None,
            database_url=REMOTE_DATABASE_URL,
            embedding_dimension=12,
            **BASE_SETTINGS,
        )


def test_secret_fields_are_redacted_in_string_representations() -> None:
    apify_secret = "private-apify-placeholder"
    openai_secret = "private-openai-placeholder"
    settings = Settings(
        _env_file=None,
        database_url=REMOTE_DATABASE_URL,
        apify_api_token=SecretStr(apify_secret),
        openai_api_key=SecretStr(openai_secret),
        **BASE_SETTINGS,
    )
    for rendered in (repr(settings), str(settings)):
        assert apify_secret not in rendered
        assert openai_secret not in rendered


def test_production_rejects_fake_provider_without_leaking_inputs() -> None:
    password = "private-password-placeholder"
    token = "private-token-placeholder"
    dsn = f"postgresql+psycopg://user:{password}@db.example.invalid/app"
    with pytest.raises(ValidationError) as caught:
        Settings(
            _env_file=None,
            app_environment="production",
            database_url=dsn,
            embedding_provider="fake",
            apify_api_token=SecretStr(token),
            **BASE_SETTINGS,
        )
    message = str(caught.value)
    assert "production-grade provider" in message
    assert password not in message
    assert token not in message
    assert dsn not in message


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_production_rejects_local_database_without_leaking_dsn(host: str) -> None:
    password = "private-password-placeholder"
    dsn = f"postgresql+psycopg://user:{password}@{host}:9999/app"
    with pytest.raises(ValidationError) as caught:
        Settings(
            _env_file=None,
            app_environment="production",
            database_url=dsn,
            embedding_provider="openai",
            openai_api_key=SecretStr("placeholder-key"),
            **BASE_SETTINGS,
        )
    message = str(caught.value)
    assert "remotely reachable database host" in message
    assert password not in message
    assert dsn not in message


def test_unknown_embedding_provider_is_rejected() -> None:
    with pytest.raises(ValidationError, match="either 'fake' or 'openai'"):
        Settings(
            _env_file=None,
            database_url=REMOTE_DATABASE_URL,
            embedding_provider="unknown",
            **BASE_SETTINGS,
        )


def test_invalid_synthesizer_reasoning_effort_is_rejected() -> None:
    with pytest.raises(
        ValidationError, match="APP_SYNTHESIZER_REASONING_EFFORT.*low.*medium.*high"
    ):
        Settings(
            _env_file=None,
            database_url=REMOTE_DATABASE_URL,
            synthesizer_reasoning_effort="extreme",
            **BASE_SETTINGS,
        )


def test_unset_synthesizer_reasoning_effort_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APP_SYNTHESIZER_REASONING_EFFORT", raising=False)

    settings = Settings(
        _env_file=None, database_url=REMOTE_DATABASE_URL, **BASE_SETTINGS
    )

    assert settings.synthesizer_reasoning_effort is None


@pytest.mark.parametrize("value", ["", "  \t  "])
def test_blank_synthesizer_reasoning_effort_normalizes_to_none(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("APP_SYNTHESIZER_REASONING_EFFORT", value)

    settings = Settings(
        _env_file=None, database_url=REMOTE_DATABASE_URL, **BASE_SETTINGS
    )

    assert settings.synthesizer_reasoning_effort is None


@pytest.mark.parametrize("value", ["low", "medium", "high"])
def test_valid_synthesizer_reasoning_effort_is_preserved(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("APP_SYNTHESIZER_REASONING_EFFORT", value)

    settings = Settings(
        _env_file=None, database_url=REMOTE_DATABASE_URL, **BASE_SETTINGS
    )

    assert settings.synthesizer_reasoning_effort == value


def test_invalid_nonempty_synthesizer_reasoning_effort_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_SYNTHESIZER_REASONING_EFFORT", "turbo")

    with pytest.raises(
        ValidationError, match="APP_SYNTHESIZER_REASONING_EFFORT.*low.*medium.*high"
    ):
        Settings(_env_file=None, database_url=REMOTE_DATABASE_URL, **BASE_SETTINGS)


def test_production_openai_requires_key_without_leaking_inputs() -> None:
    password = "private-password-placeholder"
    token = "private-token-placeholder"
    dsn = f"postgresql+psycopg://user:{password}@db.example.invalid/app"
    with pytest.raises(ValidationError) as caught:
        Settings(
            _env_file=None,
            app_environment="production",
            database_url=dsn,
            embedding_provider="openai",
            apify_api_token=SecretStr(token),
            **BASE_SETTINGS,
        )
    message = str(caught.value)
    assert "APP_OPENAI_API_KEY" in message
    assert password not in message
    assert token not in message
    assert dsn not in message


def test_settings_accept_field_names() -> None:
    settings = Settings(
        _env_file=None,
        app_name="Field Name App",
        app_environment="test",
        app_host="field-name.invalid",
        app_port=4321,
        app_reload=False,
        database_url=REMOTE_DATABASE_URL,
        database_echo=True,
    )
    assert Settings.model_config.get("populate_by_name") is True
    assert settings.app_name == "Field Name App"
    assert settings.app_host == "field-name.invalid"
    assert settings.app_port == 4321
    assert settings.app_reload is False
    assert settings.database_echo is True


def test_host_and_port_have_defaults() -> None:
    settings = Settings(_env_file=None, database_url=REMOTE_DATABASE_URL)
    assert settings.app_host == "0.0.0.0"
    assert settings.app_port == 8000


def test_import_main_without_env_file_succeeds(tmp_path: Path) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=tmp_path,
        env={
            "PYTHONPATH": str(backend_root),
            "APP_DATABASE_URL": REMOTE_DATABASE_URL,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
