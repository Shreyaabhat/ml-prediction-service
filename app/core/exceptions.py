"""Custom exceptions and the handlers that turn them into consistent JSON errors.

Every error response has this shape:
    {"error": {"code": "...", "message": "...", "request_id": "...", "details": [...]}}
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("app.errors")


class ModelNotLoadedError(Exception):
    """The model artifact isn't loaded, so we can't serve predictions."""


class InferenceError(Exception):
    """The model failed while producing a prediction."""


class DatabaseUnavailableError(Exception):
    """The database could not be reached, or the operation timed out."""


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
) -> JSONResponse:
    error: dict = {
        "code": code,
        "message": message,
        "request_id": getattr(request.state, "request_id", None),
    }

    if details:
        error["details"] = details

    return JSONResponse(
        status_code=status_code,
        content={"error": error},
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Handle request validation errors."""

    details = [
        {
            "field": ".".join(str(part) for part in err["loc"] if part != "body"),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]

    return _error_response(
        request,
        422,
        "validation_error",
        "Request validation failed",
        details,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    """Handle HTTP errors such as 404 and 405."""

    return _error_response(
        request,
        exc.status_code,
        "http_error",
        str(exc.detail),
    )


async def model_not_loaded_handler(
    request: Request,
    exc: ModelNotLoadedError,
):
    """Handle cases where the ML model isn't available."""

    return _error_response(
        request,
        503,
        "model_unavailable",
        "Model is not available. Try again later.",
    )


async def inference_error_handler(
    request: Request,
    exc: InferenceError,
):
    """Handle prediction/inference failures."""

    logger.error("inference_failed", exc_info=exc)

    return _error_response(
        request,
        500,
        "inference_failed",
        "Prediction failed",
    )


async def database_unavailable_handler(
    request: Request,
    exc: DatabaseUnavailableError,
):
    """Handle database connectivity/storage failures."""

    logger.error("database_unavailable", exc_info=exc)

    return _error_response(
        request,
        503,
        "database_unavailable",
        "Storage is temporarily unavailable. Try again later.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all application exception handlers."""

    app.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )

    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        ModelNotLoadedError,
        model_not_loaded_handler,
    )

    app.add_exception_handler(
        InferenceError,
        inference_error_handler,
    )

    app.add_exception_handler(
        DatabaseUnavailableError,
        database_unavailable_handler,
    )