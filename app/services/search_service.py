"""Search orchestration over existing retrievers; no HTTP or qrels dependencies."""

import math
from time import perf_counter

from app.core.config import MAX_QUERY_CHARS, MAX_TOP_K
from app.core.errors import ServiceError
from app.core.logging import log_event
from app.services.model_service import METHODS, ModelService

DESCRIPTIONS = {
    "bm25": "Lexical BM25 Okapi; higher lexical score ranks first.",
    "dense": "Normalized document/query embeddings ranked by cosine similarity.",
    "hybrid": "Equal-weight RRF: sum 1/(60 + source rank), candidate depth configured server-side.",
    "reranked": "Hybrid candidates reordered by cross-encoder raw logits; candidate pool is bounded.",
    "router": "Exact frozen Phase 3 policy; returns strategy, rule explanation and inference-time features.",
}
SCORE_TYPES = {
    "bm25": "bm25_okapi",
    "dense": "cosine_similarity",
    "hybrid": "rrf",
    "reranked": "cross_encoder_logit",
}


class SearchService:
    def __init__(self, resources: ModelService):
        self.resources = resources

    def validate(self, query: str, method: str, top_k: int | None) -> int:
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise ServiceError(
                "invalid_query",
                "Query must contain non-whitespace text and at most 1,024 characters.",
                422,
            )
        if method not in METHODS:
            raise ServiceError("unsupported_method", "Unsupported retrieval method.", 422)
        k = self.resources.settings.default_top_k if top_k is None else top_k
        if type(k) is not int or not 1 <= k <= MAX_TOP_K:
            raise ServiceError("invalid_top_k", "top_k must be an integer from 1 to 20.", 422)
        if method == "reranked" and k > self.resources.settings.rerank_candidates:
            raise ServiceError("invalid_top_k", "top_k exceeds the configured reranking pool.", 422)
        return k

    def search(self, query: str, method: str, top_k: int | None = None) -> dict:
        k = self.validate(query, method, top_k)
        start = perf_counter()
        with self.resources.execution() as queue_ms:
            result = self.execute(query, method, k)
        result["latency"]["queue_ms"] = queue_ms
        result["latency"]["total_service_ms"] = (perf_counter() - start) * 1000
        return result

    def execute(self, query: str, method: str, top_k: int) -> dict:
        start = perf_counter()
        before = self.resources.initialization_seconds
        resource_index = len(self.resources.initialized_names)
        trace = None
        try:
            retriever = self.resources.get(method)
            if method == "router":
                trace = retriever.search_with_trace(query, top_k)
                hits = trace.results
                strategy = trace.strategy
            elif method == "reranked":
                trace = retriever.search_with_trace(query, top_k)
                hits = trace.results
                strategy = method
            else:
                hits = retriever.search(query, top_k)
                strategy = method
            corpus = self.resources.get("corpus")
            if any(h.document_id not in corpus or not math.isfinite(h.score) for h in hits):
                raise ValueError("Invalid retrieval result")
        except ServiceError:
            raise
        except Exception as exc:
            log_event("search_failed", method=method, exception_type=type(exc).__name__)
            raise ServiceError(
                "inference_failed", "Retrieval failed; inspect model/index status and retry.", 503
            ) from exc
        engine_ms = (perf_counter() - start) * 1000
        initialization_ms = (self.resources.initialization_seconds - before) * 1000
        serialize = perf_counter()
        documents = [
            {
                "document_id": h.document_id,
                "rank": h.rank,
                "score": h.score,
                "title": corpus[h.document_id].title,
                "text": corpus[h.document_id].text,
                "metadata": {"dataset": self.resources.settings.dataset_label},
            }
            for h in hits
        ]
        rerank_ms = trace.rerank_ms if trace else 0.0
        routing = None
        if method == "router":
            protocol, _ = self.resources.get("policy")
            routing = {
                "explanation": trace.explanation,
                "features": trace.features,
                "reranker_invoked": trace.reranker_invoked,
                "candidate_count": trace.candidate_count,
                "policy_record_sha256": protocol["policy_record_sha256"],
            }
        return {
            "method": method,
            "selected_strategy": strategy,
            "top_k": top_k,
            "documents": documents,
            "score_type": SCORE_TYPES[strategy],
            "ranking_explanation": DESCRIPTIONS[strategy] + " Ties break by document ID.",
            "routing": routing,
            "candidate_count": (
                trace.candidate_count if method == "router" else len(trace.candidates)
            )
            if trace
            else 0,
            "initialized_resources": self.resources.initialized_names[resource_index:],
            "latency": {
                "initialization_ms": initialization_ms,
                "retrieval_ms": max(0.0, engine_ms - initialization_ms),
                "rerank_ms": rerank_ms,
                "serialization_ms": (perf_counter() - serialize) * 1000,
                "queue_ms": 0.0,
                "total_service_ms": (perf_counter() - start) * 1000,
            },
        }
