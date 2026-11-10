import numpy as np
import pytest

from app.data.schemas import Document
from app.retrieval.base import SearchResult
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.reranker import RerankingRetriever


def hits(ids):
    return [SearchResult(did, i, 999 - i) for i, did in enumerate(ids, 1)]


class FixedRetriever:
    def __init__(self, ids):
        self.ids, self.calls = ids, []

    def search(self, query, top_k=10):
        self.calls.append((query, top_k))
        return hits(self.ids[:top_k])


class FakeScorer:
    identity = {"model_name": "mock", "revision": "v1"}

    def __init__(self):
        self.calls = []

    def score(self, pairs, batch_size):
        self.calls.append((pairs, batch_size))
        return np.array([{" A": 1, " B": 3, " C": 3}[text] for _, text in pairs])


def test_rrf_hand_calculated():
    results = reciprocal_rank_fusion([hits(["a", "b"]), hits(["b", "c"])], 60)
    assert [r.document_id for r in results] == ["b", "a", "c"]
    assert [r.score for r in results] == pytest.approx([1 / 62 + 1 / 61, 1 / 61, 1 / 62])
    assert [r.rank for r in results] == [1, 2, 3]


def test_rrf_ties_and_empty():
    assert [r.document_id for r in reciprocal_rank_fusion([hits(["b"]), hits(["a"])])] == ["a", "b"]
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([hits(["a"]), []])[0].score == 1 / 61


@pytest.mark.parametrize("constant", [0, -1, float("nan"), float("inf"), True])
def test_invalid_rrf_constant(constant):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], constant)


def test_rrf_rejects_duplicate_ids_and_bad_ranks():
    for ranking in [hits(["a", "a"]), [SearchResult("a", 0, 1)]]:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([ranking])


def test_hybrid_depth_and_schema():
    first, second = FixedRetriever(["a", "b", "c"]), FixedRetriever(["b", "c", "d"])
    hybrid = HybridRetriever(first, second, candidate_depth=2)
    assert [r.document_id for r in hybrid.search("q", 2)] == ["b", "a"]
    assert first.calls == second.calls == [("q", 2)]
    assert hybrid.search(" ") == []
    with pytest.raises(ValueError):
        hybrid.search(None)
    with pytest.raises(ValueError):
        hybrid.search("q", 0)
    with pytest.raises(ValueError):
        HybridRetriever(first, second, candidate_depth=0)


def test_reranker_candidate_pairs_scores_and_ties():
    docs = [Document(c.lower(), "", c) for c in "ABCD"]
    scorer, candidates = FakeScorer(), FixedRetriever(["c", "a", "b", "d"])
    retriever = RerankingRetriever(candidates, docs, scorer, candidate_pool=3, batch_size=2)
    trace = retriever.search_with_trace("q", 2)
    assert [r.document_id for r in trace.results] == ["b", "c"]
    assert [r.score for r in trace.results] == [3, 3]
    assert [r.document_id for r in trace.candidates] == ["c", "a", "b"]
    assert candidates.calls == [("q", 3)]
    assert scorer.calls == [([("q", " C"), ("q", " A"), ("q", " B")], 2)]
    assert trace.rerank_ms >= 0
    assert retriever.search(" ") == []
    assert len(scorer.calls) == 1


@pytest.mark.parametrize("ids", [[], ["missing"], ["a", "a"]])
def test_reranker_empty_missing_duplicate(ids):
    docs = [Document(c.lower(), "", c) for c in "ABC"]
    retriever = RerankingRetriever(FixedRetriever(ids), docs, FakeScorer(), candidate_pool=2)
    if ids:
        with pytest.raises(ValueError):
            retriever.search("q")
    else:
        assert retriever.search("q") == []


@pytest.mark.parametrize("scores", [np.array([np.nan]), np.array([[1]]), np.array([])])
def test_reranker_bad_scores(scores):
    scorer = FakeScorer()
    scorer.score = lambda *args: scores
    retriever = RerankingRetriever(
        FixedRetriever(["a"]), [Document(c, "", c) for c in "abc"], scorer, candidate_pool=1
    )
    with pytest.raises(ValueError):
        retriever.search("q")


def test_no_entire_corpus_reranking_or_invalid_pool():
    docs = [Document(c, "", c) for c in "abc"]
    for pool in [0, 3, 4, True]:
        with pytest.raises(ValueError):
            RerankingRetriever(FixedRetriever(["a"]), docs, FakeScorer(), candidate_pool=pool)
