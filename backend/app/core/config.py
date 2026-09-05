"""Application configuration loaded from environment variables."""

from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Importing app.models.review here creates review -> db -> config during startup.
# Keep this value synchronized with app.models.review.EMBEDDING_DIMENSION via tests.
_EXPECTED_EMBEDDING_DIMENSION = 1536


class Settings(BaseSettings):
    """Runtime configuration for the App Review Intelligence API.

    Values are sourced from environment variables (or a local .env file)
    using the APP_ prefix, e.g. APP_DATABASE_URL, APP_PORT.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    app_name: str = Field(
        default="App Review Intelligence API",
        validation_alias="APP_NAME",
    )
    app_environment: str = Field(
        default="development",
        validation_alias="APP_ENVIRONMENT",
    )
    app_host: str = Field(default="0.0.0.0", validation_alias="APP_HOST")
    app_port: int = Field(default=8000, validation_alias="APP_PORT")
    app_reload: bool = Field(default=True, validation_alias="APP_RELOAD")

    # These use the default env_prefix + field-name convention, resolving
    # to APP_DATABASE_URL and APP_DATABASE_ECHO.
    database_url: str
    database_echo: bool = False

    apify_api_token: SecretStr | None = None
    apify_dataset_id: str | None = None
    embedding_provider: str = "fake"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 128
    openai_api_key: SecretStr | None = None

    # Question interpretation and answer synthesis. These reuse openai_api_key rather
    # than introducing a second credential.
    llm_provider: str = "fake"
    planner_model: str = "gpt-5-mini"
    synthesizer_model: str = "gpt-5"
    llm_timeout_seconds: int = 60
    # Reasoning models bill their internal reasoning against this same budget, and
    # reasoning runs before any output token is emitted. A budget sized only for the
    # visible answer is spent entirely on reasoning and returns empty content.
    llm_max_output_tokens: int = 8000

    @model_validator(mode="after")
    def validate_embedding_and_production_posture(self) -> "Settings":
        """Reject incompatible embedding settings and unsafe production defaults."""
        if self.embedding_dimension != _EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                "APP_EMBEDDING_DIMENSION must be 1536 to match the database vector column"
            )

        if self.embedding_provider not in {"fake", "openai"}:
            raise ValueError(
                "APP_EMBEDDING_PROVIDER must be either 'fake' or 'openai'"
            )

        if self.llm_provider not in {"fake", "openai"}:
            raise ValueError("APP_LLM_PROVIDER must be either 'fake' or 'openai'")

        if self.app_environment.lower() not in {"development", "test"}:
            # Checked first: pointing production at a local database is the more
            # fundamental misconfiguration, and reporting it before provider choices
            # keeps the message actionable.
            hostname = urlsplit(self.database_url).hostname
            if hostname is not None and hostname.lower() in {"localhost", "127.0.0.1"}:
                raise ValueError(
                    "APP_DATABASE_URL must use a remotely reachable database host "
                    "outside development and test"
                )

            if self.embedding_provider == "fake":
                raise ValueError(
                    "APP_EMBEDDING_PROVIDER must select a production-grade provider "
                    "outside development and test"
                )
            if self.embedding_provider == "openai" and self.openai_api_key is None:
                raise ValueError(
                    "APP_OPENAI_API_KEY is required when APP_EMBEDDING_PROVIDER=openai "
                    "outside development and test"
                )
            # The stub answer generator is a test fixture, not a weaker model: it must
            # never be mistaken for a real answer outside development.
            if self.llm_provider == "fake":
                raise ValueError(
                    "APP_LLM_PROVIDER must select a production-grade provider "
                    "outside development and test"
                )
            if self.llm_provider == "openai" and self.openai_api_key is None:
                raise ValueError(
                    "APP_OPENAI_API_KEY is required when APP_LLM_PROVIDER=openai "
                    "outside development and test"
                )

        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance for the process lifetime."""
    return Settings()
