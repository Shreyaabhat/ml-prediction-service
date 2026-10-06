"""Request-ID assignment and per-request logging."""
import logging
import re
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("app.request")

# Only accept client-supplied IDs that look safe; otherwise generate our own.
# (Unbounded, unvalidated header values in logs invite log injection.)
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9\-]{1,64}")


def _get_request_id(request: Request) -> str:
    incoming = request.headers.get("X-Request-ID", "")
    return incoming if _VALID_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex


async def request_logging_middleware(request: Request, call_next):
    request_id = _get_request_id(request)
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # Last line of defense: log the full stack trace, return a generic 500.
        logger.exception(
            "unhandled_exception",
            extra={
                "fields": {
                    "request_id": request_id,
                    "path": request.url.path,
                }
            },
        )
        response = JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "request_id": request_id,
                }
            },
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request_completed",
        extra={
            "fields": {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }
        },
    )

    return response