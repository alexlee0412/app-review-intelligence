"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    database_url: str = (
        "postgresql+psycopg://app_review:app_review@localhost:5432/app_review_intelligence"
    )
    database_echo: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance for the process lifetime."""
    return Settings()
