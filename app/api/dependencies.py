"""FastAPI dependencies: how routes receive their collaborators."""
from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.db.repository import PredictionRepository
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


def get_db_session(request: Request) -> Iterator[Session]:
    """One session per request, always closed (which returns its connection to the pool)."""
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_prediction_repository(session: Session = Depends(get_db_session)) -> PredictionRepository:
    return PredictionRepository(session)


def get_prediction_service(
    model_service: ModelService = Depends(get_model_service),
    repository: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionService:
    return PredictionService(model_service, repository)