"""All SQL access lives here. The rest of the app never writes queries."""
import uuid

from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session

from app.core.exceptions import DatabaseUnavailableError
from app.db.models import Prediction

# Infrastructure problems (can't connect, query timeout, pool exhausted) -> 503.
# Anything else (e.g. a constraint violation) is a bug and should surface as a 500.
_TRANSIENT_ERRORS = (OperationalError, InterfaceError, PoolTimeoutError)


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, prediction: Prediction) -> Prediction:
        try:
            self._session.add(prediction)
            self._session.commit()
        except _TRANSIENT_ERRORS as exc:
            self._session.rollback()
            raise DatabaseUnavailableError("Could not save prediction") from exc
        return prediction

    def get(self, prediction_id: uuid.UUID) -> Prediction | None:
        try:
            return self._session.get(Prediction, prediction_id)
        except _TRANSIENT_ERRORS as exc:
            self._session.rollback()
            raise DatabaseUnavailableError("Could not read prediction") from exc