"""Dependencies used by the review search API."""

import logging

from fastapi import HTTPException, status
from pydantic import ValidationError
from pydantic_settings import SettingsError

from app.schemas.search import QueryEmbedder

logger = logging.getLogger(__name__)


def get_query_embedder() -> QueryEmbedder:
    """Resolve the provider lazily when the search dependency is requested."""
    from app.core.config import get_settings
    from app.services.embedding_service import EmbeddingConfigurationError
    from app.services.embedding_service import build_embedding_provider

    try:
        provider = build_embedding_provider(get_settings())
    except (EmbeddingConfigurationError, SettingsError, ValidationError) as exc:
        logger.error("Embedding provider unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query embedding service unavailable",
        ) from exc
    return QueryEmbedder(
        name=provider.name,
        is_production_grade=provider.is_production_grade,
        embed=provider.embed_query,
    )
