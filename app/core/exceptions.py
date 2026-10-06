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
    return JSONResponse(status_code=status_code, content={"error": error})


async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Report which field failed and why, but never echo the submitted input back.
    details = [
        {
            "field": ".".join(str(part) for part in err["loc"] if part != "body"),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return _error_response(
        request, 422, "validation_error", "Request validation failed", details
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Covers 404 (unknown path), 405 (wrong method), and any HTTPException we raise.
    return _error_response(request, exc.status_code, "http_error", str(exc.detail))


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return _error_response(
        request, 503, "model_unavailable", "Model is not available. Try again later."
    )


async def inference_error_handler(request: Request, exc: InferenceError):
    logger.error("inference_failed", exc_info=exc)
    return _error_response(request, 500, "inference_failed", "Prediction failed")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
    app.add_exception_handler(InferenceError, inference_error_handler)