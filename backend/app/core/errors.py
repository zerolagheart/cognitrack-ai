"""Domain exceptions and their HTTP mapping (consistent JSON error envelope)."""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("cognitrack.api")


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"


class UnavailableError(AppError):
    """A dependency (trained model, dataset, checkpoint) is not available yet."""

    status_code = 503
    code = "unavailable"


class PayloadTooLarge(AppError):
    status_code = 413
    code = "payload_too_large"


def _envelope(code: str, message: str, request: Request, detail: dict | list | None = None) -> dict:
    return {"error": {"code": code, "message": message, "detail": detail or {}, "request_id": getattr(request.state, "request_id", None)}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(_envelope(exc.code, exc.message, request, exc.detail), status_code=exc.status_code)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(_envelope("http_error", str(exc.detail), request), status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(_envelope("request_validation", "Request validation failed", request, errors), status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        ref = str(uuid.uuid4())[:8]
        log.exception("unhandled error", extra={"ctx": {"ref": ref, "path": request.url.path}})
        return JSONResponse(_envelope("internal_error", f"Internal error (ref {ref})", request), status_code=500)
