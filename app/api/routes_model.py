from fastapi import APIRouter, Depends

from app.api.dependencies import get_model_service
from app.schemas.model import ModelInfoResponse
from app.services.model_service import ModelService

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(service: ModelService = Depends(get_model_service)) -> ModelInfoResponse:
    m = service.metadata  # raises ModelNotLoadedError -> 503 if not loaded
    return ModelInfoResponse(
        model_version=m["model_version"],
        trained_at=m["trained_at"],
        num_intents=m["num_intents"],
        confidence_threshold=m["confidence_threshold"],
        dataset_name=m["dataset"]["name"],
        scikit_learn_version=m["environment"]["scikit_learn"],
        test_metrics=m["test_metrics"],
    )