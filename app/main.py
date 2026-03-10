import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.middleware.killswitch import KillswitchMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.api.routes import admin, health, verify
from app.config import get_settings
from app.db.database import init_db, set_db_path

logger = logging.getLogger("citeguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # -- Startup --
    settings = get_settings()
    set_db_path(settings.database_url)
    await init_db()
    logger.info("Database initialized at %s", settings.database_path)

    yield

    # -- Shutdown --
    logger.info("Shutting down citeguard")


def create_app() -> FastAPI:
    app = FastAPI(
        title="citeguard",
        description="LLM hallucination detection pipeline for verifying bibliographic references",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(LoggingMiddleware)
    app.add_middleware(KillswitchMiddleware)

    # -- Routes --
    app.include_router(health.router)
    app.include_router(verify.router)
    app.include_router(admin.router)

    return app


app = create_app()