"""FastAPI application for the web backend.

Startup order matters:

1. Settings, which validates the bootstrap admin password and the LiteLLM
   master key and raises rather than starting half-configured.
2. Database, then the bootstrap admin -- created once, flagged
   must_change_password, and never silently reset on a later restart.
3. Upstream clients, held for the process lifetime.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Final

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from pii_api.admin.routes import router as admin_router
from pii_api.api.routes import router as auth_router
from pii_api.auth.service import bootstrap_admin
from pii_api.chat.routes import router as chat_router
from pii_api.db.session import Database
from pii_api.logging_config import configure_logging
from pii_api.settings import Settings, get_settings
from pii_api.upstream import LiteLlmClient, PiiServiceClient

__all__ = ["create_app", "get_app"]

VERSION: Final = "0.1.0"

logger: Final = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None, *, database: Database | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    db = database or Database(settings.database_url)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with db.session() as session:
            await bootstrap_admin(session, settings)
        logger.info("pii-api.started", version=VERSION)
        try:
            yield
        finally:
            await app.state.pii.aclose()
            await app.state.litellm.aclose()
            await db.dispose()
            logger.info("pii-api.stopped")

    app = FastAPI(
        title="PII chat and admin backend",
        version=VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.pii = PiiServiceClient(settings)
    app.state.litellm = LiteLlmClient(settings)

    @app.middleware("http")
    async def db_session_middleware(request: Request, call_next: object) -> Response:
        """One database session per request.

        A middleware rather than a `yield` dependency because the chat
        endpoint returns a StreamingResponse: with a yield-dependency the
        session closes when the handler returns, which is *before* the stream
        body runs, and every query inside the generator would hit a closed
        session.
        """
        async with db.session() as session:
            request.state.db = session
            return await call_next(request)  # type: ignore[operator, no-any-return]

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        """Liveness. Never touches the database or an upstream."""
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz(request: Request) -> JSONResponse:
        reachable = True
        try:
            await request.app.state.pii.policy()
        except Exception:
            reachable = False
        return JSONResponse(
            {
                "status": "ok" if reachable else "degraded",
                "version": VERSION,
                "pii_service_reachable": reachable,
            }
        )

    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(chat_router)
    return app


def get_app() -> FastAPI:
    return create_app()
