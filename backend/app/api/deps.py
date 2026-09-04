"""Dependencies used by the review search API."""

from fastapi import HTTPException, status

from app.schemas.search import QueryEmbedder


def get_query_embedder() -> QueryEmbedder:
    """Resolve the provider lazily when the search dependency is requested."""
    try:
        from app.core.config import get_settings
        from app.services.embedding_service import build_embedding_provider

        provider = build_embedding_provider(get_settings())
        return QueryEmbedder(
            name=provider.name,
            is_production_grade=provider.is_production_grade,
            embed=provider.embed_query,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query embedding service unavailable",
        ) from exc
