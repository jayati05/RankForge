"""Public HTTP contracts; strict bounded inputs and finite typed outputs."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import MAX_QUERY_CHARS, MAX_TOP_K


class Method(str, Enum):
    bm25 = "bm25"
    dense = "dense"
    hybrid = "hybrid"
    reranked = "reranked"
    router = "router"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SearchRequest(Contract):
    query: Annotated[str, Field(strict=True, min_length=1, max_length=MAX_QUERY_CHARS)]
    method: Method = Method.bm25
    top_k: Annotated[int, Field(strict=True, ge=1, le=MAX_TOP_K)] | None = None

    @field_validator("query")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must contain non-whitespace text")
        return value  # Preserve the exact query; do not rewrite retrieval input.

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "query": "What evidence links vitamin D deficiency to disease?",
                    "method": "hybrid",
                    "top_k": 10,
                }
            ]
        },
    )


class CompareRequest(Contract):
    query: Annotated[str, Field(strict=True, min_length=1, max_length=MAX_QUERY_CHARS)]
    methods: Annotated[list[Method], Field(min_length=1, max_length=5)] = list(Method)
    top_k: Annotated[int, Field(strict=True, ge=1, le=MAX_TOP_K)] | None = None
    _query_validator = field_validator("query")(SearchRequest.not_empty.__func__)

    @field_validator("methods")
    @classmethod
    def unique(cls, values: list[Method]) -> list[Method]:
        if len(set(values)) != len(values):
            raise ValueError("Methods must be unique")
        return values

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"query": "ERR_42", "methods": ["bm25", "hybrid"], "top_k": 5}]
        },
    )


class DocumentResult(Contract):
    document_id: str
    rank: int = Field(ge=1)
    score: float
    title: str
    text: str
    metadata: dict[str, str]


class Latency(Contract):
    initialization_ms: float = Field(ge=0)
    retrieval_ms: float = Field(
        ge=0,
        description="Complete pipeline time excluding lazy initialization; includes reranking where selected.",
    )
    rerank_ms: float = Field(
        ge=0,
        description="Incremental cross-encoder scoring/sorting time, a component of retrieval_ms.",
    )
    serialization_ms: float = Field(ge=0)
    queue_ms: float = Field(ge=0)
    total_service_ms: float = Field(
        ge=0,
        description="Service orchestration time; excludes HTTP JSON encoding and network transit.",
    )


class Routing(Contract):
    explanation: str
    features: dict[str, int | float | bool]
    reranker_invoked: bool
    candidate_count: int = Field(ge=0)
    policy_record_sha256: str


class MethodResult(Contract):
    method: Method
    selected_strategy: Method
    top_k: int = Field(ge=1, le=MAX_TOP_K)
    documents: list[DocumentResult]
    score_type: str
    ranking_explanation: str
    routing: Routing | None = None
    candidate_count: int = Field(
        ge=0, description="Actual reranker-input count; zero when no reranking pool is requested."
    )
    initialized_resources: list[str]
    latency: Latency


class SearchResponse(MethodResult):
    request_id: str


class Agreement(Contract):
    first: Method
    second: Method
    shared_document_ids: list[str]
    jaccard: float = Field(ge=0, le=1)
    same_position_fraction: float = Field(ge=0, le=1)


class CompareResponse(Contract):
    request_id: str
    results: list[MethodResult]
    agreement: list[Agreement]
    queue_ms: float = Field(ge=0)
    total_service_ms: float = Field(ge=0)
    interpretation: str


class ErrorDetail(Contract):
    field: str
    code: str
    message: str


class ErrorInfo(Contract):
    code: str
    message: str
    details: list[ErrorDetail] = []


class ErrorResponse(Contract):
    request_id: str
    error: ErrorInfo


class ResourceStatus(Contract):
    state: Literal["not_loaded", "initializing", "ready", "failed", "released"]
    initialization_ms: float = Field(ge=0)
    error_code: str | None
    initialization_attempts: int = Field(default=0, ge=0)
    initialization_count: int = Field(default=0, ge=0)


class ActivityStatus(Contract):
    active: int = Field(ge=0, le=1)
    queued: int = Field(ge=0)
    stopping: bool


class DatasetStatus(Contract):
    name: str
    available: bool
    loaded: bool


class IndexStatus(Contract):
    state: Literal["not_loaded", "initializing", "ready", "failed", "released"]
    present: bool | None = None


class ModelStatus(ResourceStatus):
    identifier: str | None
    revision: str | None
    device: Literal["cpu"]
    snapshot_present: bool


class HealthResponse(Contract):
    request_id: str
    version: str
    status: Literal["ok", "degraded", "stopping", "stopped"]
    dataset: DatasetStatus
    indexes: dict[str, IndexStatus]
    resources: dict[str, ResourceStatus]
    activity: ActivityStatus
    available_methods: list[Method]
    configuration_present: dict[str, bool]
    neural_dependencies_available: bool
    availability_note: str


class ModelsResponse(Contract):
    request_id: str
    models: dict[str, ModelStatus]
    indexes: dict[str, IndexStatus]
    neural_dependencies_available: bool


class MethodDescription(Contract):
    method: Method
    description: str
    score_type: str
    available: bool


class MethodsResponse(Contract):
    request_id: str
    methods: list[MethodDescription]
    top_k_range: list[int]
    default_top_k: int
    max_query_chars: int
