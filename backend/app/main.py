"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import get_settings
from backend.app.core.errors import register_error_handlers
from backend.app.database.session import init_db
from ml.utils import configure_logging

log = logging.getLogger("cognitrack.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.auth_enabled:
        if settings.environment == "production":
            raise RuntimeError("Refusing to start in production without API keys (COGNITRACK_*_API_KEYS).")
        log.warning("API authentication DISABLED (no API keys configured) - development use only")
    init_db()
    log.info("CogniTrack API started", extra={"ctx": {"environment": settings.environment, "db": settings.database_url.split("://")[0]}})
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="CogniTrack AI API",
        version="0.1.0",
        description=(
            "Multimodal longitudinal speech intelligence - **research and clinical decision-support prototype, not a medical "
            "device**. Outputs are statistical signals for review by qualified clinicians and never diagnoses."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Path only (no query strings, which might carry identifiers) and no bodies are logged.
        log.info("request", extra={"ctx": {"request_id": request.state.request_id, "method": request.method, "path": request.url.path,
                                           "status": response.status_code, "ms": round((time.perf_counter() - start) * 1000, 1)}})
        return response

    register_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
