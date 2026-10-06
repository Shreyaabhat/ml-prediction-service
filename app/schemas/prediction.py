"""Request/response contracts for POST /predict."""
from pydantic import BaseModel, ConfigDict, Field

# Bounding input size protects the service from huge payloads. Real utterances are short.
MAX_TEXT_LENGTH = 500


class PredictRequest(BaseModel):
    # Strip surrounding whitespace first, so "   " fails the min_length check.
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="The user utterance to classify.",
        examples=["set a timer for ten minutes"],
    )


class PredictResponse(BaseModel):
    intent: str = Field(description='Predicted intent, or "oos" if below the confidence threshold.')
    confidence: float = Field(ge=0.0, le=1.0, description="Model's top class probability.")
    is_out_of_scope: bool
    model_version: str