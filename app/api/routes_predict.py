from fastapi import APIRouter, Depends

from app.api.dependencies import get_model_service
from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.model_service import ModelService

router = APIRouter(tags=["prediction"])


# Plain `def` (not `async def`): inference is CPU-bound, so FastAPI runs this in its
# threadpool instead of blocking the event loop.
@router.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest, service: ModelService = Depends(get_model_service)
) -> PredictResponse:
    result = service.predict(body.text)
    return PredictResponse(
        intent=result.intent,
        confidence=result.confidence,
        is_out_of_scope=result.is_out_of_scope,
        model_version=service.version,
    )