"""Thin HTTP adapters; all resource/search/comparison logic lives in services."""

from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import resources
from app.api.examples import COMPARE_EXAMPLE, ERROR_EXAMPLE, SEARCH_EXAMPLE
from app.api.schemas import (
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    HealthResponse,
    MethodsResponse,
    ModelsResponse,
    SearchRequest,
    SearchResponse,
)
from app.core.config import MAX_QUERY_CHARS, MAX_TOP_K
from app.services.comparison_service import ComparisonService
from app.services.model_service import METHODS, ModelService
from app.services.search_service import DESCRIPTIONS, SCORE_TYPES, SearchService

router = APIRouter(
    responses={
        **{status: {"model": ErrorResponse} for status in (400, 413, 429, 500, 503)},
        422: {"model": ErrorResponse, "content": {"application/json": {"example": ERROR_EXAMPLE}}},
    }
)


@router.get("/health", response_model=HealthResponse, summary="Cheap liveness and resource status")
def health(request: Request, manager: ModelService = Depends(resources)):
    """Reports file/dependency presence and loaded state. Never loads models, builds indexes or downloads."""
    try:
        build_version = version("rankforge")
    except PackageNotFoundError:
        build_version = "unknown"
    return {"request_id": request.state.request_id, "version": build_version, **manager.snapshot()}


@router.get("/models", response_model=ModelsResponse, summary="Local model and cache status")
def models(request: Request, manager: ModelService = Depends(resources)):
    """Lists public model identifiers/revisions and local-only CPU state without exposing filesystem paths."""
    return {"request_id": request.state.request_id, **manager.models()}


@router.get(
    "/methods", response_model=MethodsResponse, summary="Supported retrieval methods and bounds"
)
def methods(request: Request, manager: ModelService = Depends(resources)):
    """Availability is provisional file/dependency presence until first use validates resources."""
    snapshot = manager.snapshot()
    return {
        "request_id": request.state.request_id,
        "methods": [
            {
                "method": name,
                "description": DESCRIPTIONS[name],
                "score_type": SCORE_TYPES.get(name, "selected_strategy_score"),
                "available": name in snapshot["available_methods"],
            }
            for name in METHODS
        ],
        "top_k_range": [1, MAX_TOP_K],
        "default_top_k": manager.settings.default_top_k,
        "max_query_chars": MAX_QUERY_CHARS,
    }


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search the configured corpus",
    responses={
        200: {
            "description": "Recorded synthetic smoke response; incidental timings are not benchmark results.",
            "content": {"application/json": {"example": SEARCH_EXAMPLE}},
        }
    },
)
def search(body: SearchRequest, request: Request, manager: ModelService = Depends(resources)):
    """Existing retrievers, read-only prepared neural resources, server-side configuration only. Lazy initialization is separately timed. Router returns its frozen-policy explanation."""
    result = SearchService(manager).search(
        body.query,
        body.method.value if "method" in body.model_fields_set else manager.settings.default_method,
        body.top_k,
    )
    return {"request_id": request.state.request_id, **result}


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare rankings, not relevance quality",
    responses={
        200: {
            "description": "Recorded one-method smoke response; request multiple methods for pairwise agreement.",
            "content": {"application/json": {"example": COMPARE_EXAMPLE}},
        }
    },
)
def compare(body: CompareRequest, request: Request, manager: ModelService = Depends(resources)):
    """Runs one to five unique methods independently on the same corpus. Returns rank overlap/order, with no Recall/nDCG or other quality claims without judgments."""
    result = ComparisonService(SearchService(manager)).compare(
        body.query, [m.value for m in body.methods], body.top_k
    )
    return {"request_id": request.state.request_id, **result}
