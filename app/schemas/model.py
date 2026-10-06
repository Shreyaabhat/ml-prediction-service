from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str
    num_intents: int
    confidence_threshold: float
    dataset_name: str
    scikit_learn_version: str
    test_metrics: dict[str, float]