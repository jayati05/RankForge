from app.data.schemas import Document
from app.evaluation.features import CorpusFeatures
from app.retrieval.base import SearchResult
from app.retrieval.reranker import RerankTrace
from app.retrieval.router import QueryAwareRouter, RoutingPolicy


def test_live_rerank_branch_records_invocation_without_labels():
    class Lexical:
        def search(self, query, top_k=10):
            raise AssertionError("Unselected retriever must not execute")

    class Rerank:
        def search_with_trace(self, query, top_k=10):
            hits = [SearchResult("a", 1, 3.0)]
            return RerankTrace(hits, hits, 1.5, hits)

    corpus = CorpusFeatures([Document("a", "", "alpha"), Document("b", "", "beta")])
    router = QueryAwareRouter(
        corpus,
        {"bm25": Lexical(), "hybrid": Lexical(), "reranked": Rerank()},
        RoutingPolicy(1.01, 8),
    )
    trace = router.search_with_trace("alpha", 1)
    assert trace.strategy == "reranked"
    assert trace.reranker_invoked and trace.candidate_count == 1
    assert trace.rerank_ms == 1.5
    assert "token count 1 <= 8" in trace.explanation
    assert trace.routing_overhead_ms >= 0
    assert trace.results[0].document_id == "a"
