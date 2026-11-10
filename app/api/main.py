"""Application factory and controlled lifespan; importing this module loads no models."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from app.api.errors import error_response
from app.api.middleware import BoundaryMiddleware
from app.api.routes import router
from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.logging import configure_logging, log_event
from app.services.model_service import ModelService


def create_app(settings: Settings | None = None, resource_factory=ModelService) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application):
        configure_logging(settings.log_level)
        application.state.resources = resource_factory(settings)
        log_event("service_started", device="cpu", lazy_initialization=True)
        try:
            yield
        finally:
            await run_in_threadpool(application.state.resources.close)
            log_event("service_stopped")

    application = FastAPI(
        title="RankForge Local Search API",
        version="0.1.0",
        lifespan=lifespan,
        description="Local research service over existing retrievers. Models/indexes load lazily, never download through HTTP. Compare reports ranking agreement, not judged relevance quality.",
        docs_url="/docs",
        redoc_url="/redoc",
        debug=False,
    )
    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    application.add_middleware(BoundaryMiddleware, max_body_bytes=settings.max_body_bytes)

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return error_response(exc.status_code, request.state.request_id, exc.code, exc.message)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        details = [
            {
                "field": ".".join(map(str, error["loc"])),
                "code": error["type"],
                "message": error["msg"],
            }
            for error in exc.errors()
        ]
        return error_response(
            422, request.state.request_id, "invalid_request", "Request validation failed.", details
        )

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return error_response(
            exc.status_code,
            request.state.request_id,
            "http_error",
            {404: "Endpoint not found.", 405: "HTTP method not allowed."}.get(
                exc.status_code, "HTTP request failed."
            ),
        )

    application.include_router(router)
    return application


app = create_app()
