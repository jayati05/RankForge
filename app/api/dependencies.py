from fastapi import Request

from app.services.model_service import ModelService


def resources(request: Request) -> ModelService:
    return request.app.state.resources
