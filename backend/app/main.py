"""FastAPI application factory and entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ask import router as ask_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
    )

    if settings.cors_allow_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["POST"],
            allow_headers=["Accept", "Content-Type"],
        )

    application.include_router(health_router)
    application.include_router(search_router)
    application.include_router(ask_router)

    return application


app = create_app()
