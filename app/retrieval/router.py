"""Transparent routing from corpus/query lexical features, with no qrels access."""

from dataclasses import asdict, dataclass
from time import perf_counter

from app.evaluation.features import CorpusFeatures, QueryFeatures
from app.retrieval.base import Retriever, SearchResult, validate_top_k


@dataclass(frozen=True)
class RoutingPolicy:
    bm25_rare_min: float = 0.25
    rerank_max_tokens: int = 8
    bm25_coverage_min: float = 0.8

    def __post_init__(self) -> None:
        if not 0 <= self.bm25_rare_min <= 1.01 or not 0 <= self.bm25_coverage_min <= 1:
            raise ValueError("Invalid policy proportions")
        if type(self.rerank_max_tokens) is not int or self.rerank_max_tokens < 0:
            raise ValueError("Invalid rerank token threshold")

    def decide(self, features: QueryFeatures) -> tuple[str, str]:
        if features.token_count == 0:
            return "bm25", "No lexical tokens; choose the bounded empty lexical search"
        if (
            features.rare_term_proportion >= self.bm25_rare_min
            and features.corpus_term_coverage >= self.bm25_coverage_min
        ):
            return "bm25", (
                f"Rare-term proportion {features.rare_term_proportion:.3f} >= {self.bm25_rare_min}; "
                f"coverage {features.corpus_term_coverage:.3f} >= {self.bm25_coverage_min}"
            )
        if features.token_count <= self.rerank_max_tokens:
            return (
                "reranked",
                f"Lexical shortcut not met; token count {features.token_count} <= {self.rerank_max_tokens}",
            )
        return (
            "hybrid",
            f"Lexical shortcut not met; token count {features.token_count} > {self.rerank_max_tokens}; use RRF",
        )


@dataclass
class RoutingTrace:
    results: list[SearchResult]
    strategy: str
    explanation: str
    features: dict
    reranker_invoked: bool
    candidate_count: int
    latency_ms: float
    routing_overhead_ms: float
    rerank_ms: float


class QueryAwareRouter:
    def __init__(
        self, corpus_features: CorpusFeatures, systems: dict[str, Retriever], policy: RoutingPolicy
    ):
        if not {"bm25", "hybrid", "reranked"} <= set(systems):
            raise ValueError("Router requires bm25, hybrid and reranked systems")
        self.corpus_features, self.systems, self.policy = corpus_features, systems, policy

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        return self.search_with_trace(query, top_k).results

    def search_with_trace(self, query: str, top_k: int = 10) -> RoutingTrace:
        validate_top_k(top_k)
        start = perf_counter()
        features = self.corpus_features.extract(query)
        strategy, explanation = self.policy.decide(features)
        overhead = (perf_counter() - start) * 1000
        candidate_count, rerank_ms = 0, 0.0
        if strategy == "reranked":
            trace = self.systems[strategy].search_with_trace(query, top_k)
            results, candidate_count, rerank_ms = (
                trace.results,
                len(trace.candidates),
                trace.rerank_ms,
            )
        else:
            results = self.systems[strategy].search(query, top_k)
        return RoutingTrace(
            results,
            strategy,
            explanation,
            asdict(features),
            strategy == "reranked",
            candidate_count,
            (perf_counter() - start) * 1000,
            overhead,
            rerank_ms,
        )
