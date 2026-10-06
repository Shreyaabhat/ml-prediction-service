"""Orchestrates one prediction: run the model, then persist the metadata.

In Phase 4 the cache lookup is added here. Routes stay thin.
"""
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass

from app.core.exceptions import DatabaseUnavailableError, InferenceError
from app.db.models import Prediction
from app.db.repository import PredictionRepository
from app.ml.preprocess import normalize_text
from app.services.model_service import ModelService, PredictionResult

logger = logging.getLogger("app.prediction")


@dataclass(frozen=True)
class PredictionOutcome:
    prediction_id: uuid.UUID
    result: PredictionResult
    model_version: str
    latency_ms: float


def _text_hash(text: str) -> str:
    # Hash the NORMALIZED text, so "Set a Timer" and "set a  timer" hash identically.
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class PredictionService:
    def __init__(self, model_service: ModelService, repository: PredictionRepository) -> None:
        self._model_service = model_service
        self._repository = repository

    def predict(self, text: str) -> PredictionOutcome:
        prediction_id = uuid.uuid4()
        start = time.perf_counter()

        try:
            result = self._model_service.predict(text)
        except InferenceError:
            self._record_failure(prediction_id, text, _elapsed_ms(start), "inference_failed")
            raise

        latency_ms = _elapsed_ms(start)
        self._repository.create(
            Prediction(
                id=prediction_id,
                model_version=self._model_service.version,
                input_text=text,
                input_sha256=_text_hash(text),
                predicted_intent=result.intent,
                confidence=result.confidence,
                is_out_of_scope=result.is_out_of_scope,
                latency_ms=latency_ms,
                status="success",
            )
        )
        return PredictionOutcome(
            prediction_id=prediction_id,
            result=result,
            model_version=self._model_service.version,
            latency_ms=latency_ms,
        )

    def _record_failure(
        self, prediction_id: uuid.UUID, text: str, latency_ms: float, error_code: str
    ) -> None:
        """Best effort: never let a failed audit write hide the ORIGINAL error."""
        try:
            self._repository.create(
                Prediction(
                    id=prediction_id,
                    model_version=self._model_service.version,
                    input_text=text,
                    input_sha256=_text_hash(text),
                    latency_ms=latency_ms,
                    status="failed",
                    error_code=error_code,
                )
            )
        except DatabaseUnavailableError:
            logger.warning(
                "failed_prediction_not_recorded",
                extra={"fields": {"prediction_id": str(prediction_id)}},
            )