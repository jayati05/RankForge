"""Cross-encoder reranking of a bounded hybrid candidate pool."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from app.data.schemas import Document
from app.retrieval.base import Retriever, SearchResult, validate_top_k
from app.retrieval.neural import PairScorer, document_text, positive_integer


@dataclass
class RerankTrace:
    results: list[SearchResult]
    candidates: list[SearchResult]
    rerank_ms: float
    reranked_candidates: list[SearchResult] = field(default_factory=list)


class RerankingRetriever:
    def __init__(
        self,
        hybrid: Retriever,
        documents: Iterable[Document],
        scorer: PairScorer,
        candidate_pool: int = 20,
        batch_size: int = 16,
    ):
        positive_integer(candidate_pool, "candidate_pool")
        positive_integer(batch_size, "batch_size")
        docs = list(documents)
        if not docs or len({d.id for d in docs}) != len(docs):
            raise ValueError("Corpus must be nonempty with unique IDs")
        if candidate_pool >= len(docs):
            raise ValueError("Rerank candidate pool must be smaller than the corpus")
        self.documents = {d.id: d for d in docs}
        self.hybrid, self.scorer = hybrid, scorer
        self.candidate_pool, self.batch_size = candidate_pool, batch_size

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        return self.search_with_trace(query, top_k).results

    def search_with_trace(self, query: str, top_k: int = 10) -> RerankTrace:
        validate_top_k(top_k)
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not query.strip():
            return RerankTrace([], [], 0.0)
        candidates = self.hybrid.search(query, self.candidate_pool)
        ids = [r.document_id for r in candidates]
        if len(ids) > self.candidate_pool or len(ids) != len(set(ids)):
            raise ValueError("Candidate retriever exceeded its pool or returned duplicates")
        if any(did not in self.documents for did in ids):
            raise ValueError("Candidate references missing document")
        if not ids:
            return RerankTrace([], [], 0.0)
        pairs = [
            (query, document_text(self.documents[did].title, self.documents[did].text))
            for did in ids
        ]
        start = perf_counter()
        scores = np.asarray(self.scorer.score(pairs, self.batch_size))
        if scores.shape != (len(ids),) or not np.isfinite(scores).all():
            raise ValueError("Cross-encoder returned wrong shape or nonfinite scores")
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
        results = [SearchResult(ids[i], rank, float(scores[i])) for rank, i in enumerate(order, 1)]
        return RerankTrace(results[:top_k], candidates, (perf_counter() - start) * 1000, results)
