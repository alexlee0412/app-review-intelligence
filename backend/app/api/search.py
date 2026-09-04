"""HTTP endpoint for evidence-backed semantic review retrieval."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_query_embedder
from app.core.db import get_db
from app.schemas.search import QueryEmbedder, SearchRequest, SearchResponse
from app.services.review_search import QueryEmbeddingError, search_reviews

router = APIRouter(prefix="/api/v1/reviews", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def post_search(
    request: SearchRequest,
    session: Session = Depends(get_db),
    embedder: QueryEmbedder = Depends(get_query_embedder),
) -> SearchResponse:
    """Return filtered semantic matches together with evidence and query trace."""
    try:
        return search_reviews(session, request, embedder)
    except QueryEmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query embedding service unavailable",
        ) from exc
