"""Loads the model artifact and runs inference. Knows nothing about HTTP."""
import json
import logging
from dataclasses import dataclass

import joblib
import sklearn

from app.core.exceptions import InferenceError, ModelNotLoadedError
from app.ml.evaluate import top_prediction
from app.ml.registry import model_dir

logger = logging.getLogger("app.model")

OOS_INTENT = "oos"


@dataclass(frozen=True)
class PredictionResult:
    intent: str
    confidence: float
    is_out_of_scope: bool


class ModelService:
    def __init__(self, version: str) -> None:
        self.version = version
        self._model = None
        self._metadata: dict | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            raise ModelNotLoadedError()
        return self._metadata

    def load(self) -> None:
        directory = model_dir(self.version)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        model = joblib.load(directory / "model.joblib")

        trained_with = metadata.get("environment", {}).get("scikit_learn")
        if trained_with != sklearn.__version__:
            logger.warning(
                "sklearn_version_mismatch",
                extra={"fields": {"trained_with": trained_with, "running": sklearn.__version__}},
            )

        # Assign only after BOTH loads succeed, so we never end up half-loaded.
        self._metadata = metadata
        self._model = model

    def predict(self, text: str) -> PredictionResult:
        if self._model is None or self._metadata is None:
            raise ModelNotLoadedError()
        try:
            labels, confidences = top_prediction(self._model, [text])
        except Exception as exc:
            raise InferenceError("Model inference failed") from exc

        confidence = float(confidences[0])
        is_oos = confidence < self._metadata["confidence_threshold"]
        return PredictionResult(
            intent=OOS_INTENT if is_oos else str(labels[0]),
            confidence=round(confidence, 4),
            is_out_of_scope=is_oos,
        )