"""Equal-weight Reciprocal Rank Fusion; raw retrieval scores are never added."""

import math
from collections.abc import Sequence

from app.retrieval.base import Retriever, SearchResult, validate_top_k
from app.retrieval.neural import positive_integer


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], constant: float = 60
) -> list[SearchResult]:
    if isinstance(constant, bool) or not math.isfinite(constant) or constant <= 0:
        raise ValueError("RRF constant must be finite and positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        for expected_rank, result in enumerate(ranking, 1):
            if result.document_id in seen or result.rank != expected_rank:
                raise ValueError("Input rankings must have unique IDs and contiguous 1-based ranks")
            seen.add(result.document_id)
            scores[result.document_id] = scores.get(result.document_id, 0) + 1 / (
                constant + result.rank
            )
    ordered = sorted(scores, key=lambda did: (-scores[did], did))
    return [SearchResult(did, rank, scores[did]) for rank, did in enumerate(ordered, 1)]


class HybridRetriever:
    def __init__(
        self, bm25: Retriever, dense: Retriever, candidate_depth: int = 50, rrf_constant: float = 60
    ):
        positive_integer(candidate_depth, "candidate_depth")
        reciprocal_rank_fusion([], rrf_constant)
        self.bm25, self.dense = bm25, dense
        self.candidate_depth, self.rrf_constant = candidate_depth, rrf_constant

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        validate_top_k(top_k)
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not query.strip():
            return []
        rankings = [
            self.bm25.search(query, self.candidate_depth),
            self.dense.search(query, self.candidate_depth),
        ]
        return reciprocal_rank_fusion(rankings, self.rrf_constant)[:top_k]
