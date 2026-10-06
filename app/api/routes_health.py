
from fastapi import APIRouter, Depends, Response

from app.api.dependencies import get_model_service
from app.schemas.health import HealthResponse, ReadyResponse
from app.services.model_service import ModelService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: the process is up. Deliberately checks NO dependencies."""
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse}},
)
def ready(
    response: Response,
    service: ModelService = Depends(get_model_service),
) -> ReadyResponse:
    """Readiness: checks whether the model is loaded."""

    is_model_ready = service.is_loaded

    # For now, the database is considered OK if the application
    # successfully created the database engine during startup.
    database_ok = True

    is_ready = is_model_ready and database_ok

    if not is_ready:
        response.status_code = 503

    return ReadyResponse(
        status="ready" if is_ready else "not_ready",
        model_loaded=is_model_ready,
        database_ok=database_ok,
    )

