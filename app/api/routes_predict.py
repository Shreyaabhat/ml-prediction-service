import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_prediction_repository, get_prediction_service
from app.db.repository import PredictionRepository
from app.schemas.prediction import PredictionRecordResponse, PredictRequest, PredictResponse
from app.services.prediction_service import PredictionService

router = APIRouter(tags=["prediction"])


# Plain `def` (not `async def`): inference and the DB driver are synchronous, so
# FastAPI runs this in its threadpool instead of blocking the event loop.
@router.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest, service: PredictionService = Depends(get_prediction_service)
) -> PredictResponse:
    outcome = service.predict(body.text)
    return PredictResponse(
        prediction_id=outcome.prediction_id,
        intent=outcome.result.intent,
        confidence=outcome.result.confidence,
        is_out_of_scope=outcome.result.is_out_of_scope,
        model_version=outcome.model_version,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionRecordResponse)
def get_prediction(
    prediction_id: uuid.UUID,  # a malformed UUID is rejected with 422 before we get here
    repository: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionRecordResponse:
    record = repository.get(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return PredictionRecordResponse(
        prediction_id=record.id,
        created_at=record.created_at,
        model_version=record.model_version,
        input_text=record.input_text,
        predicted_intent=record.predicted_intent,
        confidence=record.confidence,
        is_out_of_scope=record.is_out_of_scope,
        latency_ms=record.latency_ms,
        status=record.status,
    )