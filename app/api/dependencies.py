from fastapi import Request

from app.services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    """Give routes the ModelService created at startup."""
    return request.app.state.model_service