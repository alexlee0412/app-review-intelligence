"""Application and database health endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.db import check_database_connection

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class DatabaseHealthResponse(BaseModel):
    status: str
    database: str


@router.get("/health", response_model=HealthResponse)
def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report application liveness. Does not touch the database."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_environment,
    )


@router.get("/health/db", response_model=DatabaseHealthResponse)
def get_health_db(
    database_reachable: bool = Depends(check_database_connection),
) -> DatabaseHealthResponse:
    """Report database reachability via a lightweight SELECT 1 query."""
    if not database_reachable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "error", "database": "unreachable"},
        )
    return DatabaseHealthResponse(status="ok", database="reachable")
