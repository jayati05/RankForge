"""Compare rank sets/order on one prepared corpus, without relevance metrics."""

from itertools import combinations
from time import perf_counter

from app.core.errors import ServiceError
from app.services.search_service import SearchService


class ComparisonService:
    def __init__(self, search: SearchService):
        self.search = search

    def compare(self, query: str, methods: list[str], top_k: int | None = None) -> dict:
        if not methods or len(methods) > 5 or len(set(methods)) != len(methods):
            raise ServiceError(
                "invalid_methods", "Choose one to five unique retrieval methods.", 422
            )
        k = None
        for method in methods:
            k = self.search.validate(query, method, top_k)
        start = perf_counter()
        with self.search.resources.execution() as queue_ms:
            results = [self.search.execute(query, method, k) for method in methods]
        pairs = []
        for a, b in combinations(results, 2):
            first = [r["document_id"] for r in a["documents"]]
            second = [r["document_id"] for r in b["documents"]]
            sa, sb = set(first), set(second)
            pairs.append(
                {
                    "first": a["method"],
                    "second": b["method"],
                    "shared_document_ids": sorted(sa & sb),
                    "jaccard": len(sa & sb) / len(sa | sb) if sa | sb else 0.0,
                    "same_position_fraction": sum(x == y for x, y in zip(first, second))
                    / max(len(first), len(second))
                    if first or second
                    else 0.0,
                }
            )
        return {
            "results": results,
            "agreement": pairs,
            "queue_ms": queue_ms,
            "total_service_ms": (perf_counter() - start) * 1000,
            "interpretation": "Rank overlap/order only. Scores from different methods are not directly comparable; no relevance metrics are computed.",
        }
