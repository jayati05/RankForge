import math
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from app.data.preprocessing import tokenize
from app.data.schemas import Document
from app.retrieval.base import SearchResult, validate_top_k


class BM25Retriever:
    def __init__(self, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Require finite k1 > 0 and 0 <= b <= 1")
        docs = sorted(documents, key=lambda doc: doc.id)
        if not docs or len({doc.id for doc in docs}) != len(docs):
            raise ValueError("Corpus must be nonempty with unique IDs")
        self.ids = [doc.id for doc in docs]
        tokens = [tokenize(doc.title + " " + doc.text) for doc in docs]
        if not any(tokens):
            raise ValueError("Corpus has no indexable tokens")
        self.vocabulary = {token for row in tokens for token in row}
        self.index = BM25Okapi(tokens, k1=k1, b=b)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        validate_top_k(top_k)
        tokens = tokenize(query)
        if not tokens or not self.vocabulary.intersection(tokens):
            return []
        scores = self.index.get_scores(tokens)
        if not all(math.isfinite(float(score)) for score in scores):
            raise RuntimeError("BM25 returned nonfinite scores")
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))
        return [
            SearchResult(self.ids[i], rank, float(scores[i]))
            for rank, i in enumerate(order[:top_k], 1)
        ]
