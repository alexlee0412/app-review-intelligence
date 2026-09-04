"""Embedding providers for review retrieval.

The fake provider produces hash-seeded vectors with no semantic meaning. It exists
only to keep tests deterministic and local integration runnable, and it must never
serve production traffic.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import SecretStr

from app.core.config import Settings
from app.models.review import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    name: str
    dimension: int
    is_production_grade: bool

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingError(RuntimeError):
    """Base exception for embedding failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Embedding configuration is missing or unsafe."""


class EmbeddingDimensionError(EmbeddingError):
    """An embedding vector does not match the database dimension."""


class EmbeddingAPIError(EmbeddingError):
    """A sanitized provider request failure."""

    def __init__(self, status_code: int | None = None) -> None:
        self.status_code = status_code
        if status_code is None:
            message = "Embedding provider request failed"
        else:
            message = f"Embedding provider request failed with HTTP status {status_code}"
        super().__init__(message)


def _normalize_vector(vector: Sequence[float]) -> list[float]:
    if len(vector) != EMBEDDING_DIMENSION:
        raise EmbeddingDimensionError(
            f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {len(vector)}"
        )
    values = [float(value) for value in vector]
    magnitude = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(magnitude) or magnitude == 0:
        raise EmbeddingError("Embedding vector must have a finite, non-zero L2 norm")
    return [value / magnitude for value in values]


class FakeEmbeddingProvider:
    """Deterministic offline vectors used only in development and tests."""

    name = "fake"
    dimension = EMBEDDING_DIMENSION
    is_production_grade = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            generator = random.Random(int.from_bytes(digest, byteorder="big"))
            vector = [
                generator.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIMENSION)
            ]
            vectors.append(_normalize_vector(vector))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider with batching and bounded retries."""

    name = "openai"
    dimension = EMBEDDING_DIMENSION
    is_production_grade = True

    def __init__(
        self,
        api_key: SecretStr | None,
        *,
        model: str = "text-embedding-3-small",
        batch_size: int = 128,
        client: Any | None = None,
    ) -> None:
        if api_key is None:
            raise EmbeddingConfigurationError(
                "APP_OPENAI_API_KEY is required when APP_EMBEDDING_PROVIDER=openai"
            )
        if batch_size < 1:
            raise EmbeddingConfigurationError(
                "APP_EMBEDDING_BATCH_SIZE must be positive"
            )
        self.model = model
        self.batch_size = batch_size
        if client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key.get_secret_value(),
                max_retries=0,
            )
        else:
            self._client = client

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response: Any | None = None
        for attempt in range(5):
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=EMBEDDING_DIMENSION,
                )
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = status_code == 429 or (
                    isinstance(status_code, int) and status_code >= 500
                )
                if not retryable or attempt == 4:
                    raise EmbeddingAPIError(status_code) from None
                time.sleep(min(0.5 * (2**attempt), 4.0))

        if response is None:
            raise EmbeddingAPIError()
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise EmbeddingError(
                "Embedding provider returned a different number of vectors than inputs"
            )
        return [_normalize_vector(item.embedding) for item in data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def build_embedding_input(title: str | None, body: str) -> str:
    """Build the single text representation shared by indexing and querying."""
    return f"{title}\n\n{body}" if title else body


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the configured provider and log only its public name."""
    if settings.embedding_provider == "fake":
        provider: EmbeddingProvider = FakeEmbeddingProvider()
    elif settings.embedding_provider == "openai":
        provider = OpenAIEmbeddingProvider(
            settings.openai_api_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    else:
        raise EmbeddingConfigurationError(
            "APP_EMBEDDING_PROVIDER must be either 'fake' or 'openai'"
        )
    logger.info("Embedding provider selected: %s", provider.name)
    return provider
