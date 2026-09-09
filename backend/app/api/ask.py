"""HTTP endpoint for grounded review question answering."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas.answer import AnswerResponse
from app.services.ask_service import answer_question
from app.services.llm_provider import (
    LLMConfigurationError,
    LLMError,
    LLMResponseError,
)
from app.services.review_search import QueryEmbeddingError

router = APIRouter(prefix="/api/v1/reviews", tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


@router.post("/ask", response_model=AnswerResponse)
def post_ask(
    request: AskRequest,
    session: Session = Depends(get_db),
) -> AnswerResponse:
    """Answer one review question through the canonical ask service."""
    try:
        return answer_question(session, request.question)
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Answer service is not configured.",
        ) from exc
    except LLMResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Answer service is temporarily unavailable.",
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Answer service is temporarily unavailable.",
        ) from exc
    except QueryEmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Review search service is temporarily unavailable.",
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Review database is temporarily unavailable.",
        ) from exc
