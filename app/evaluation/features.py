"""Inference-safe query features and optional saved-ranking diagnostics."""

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from app.data.preprocessing import tokenize
from app.data.schemas import Document
from app.retrieval.base import SearchResult


@dataclass(frozen=True)
class QueryFeatures:
    token_count: int
    unique_token_count: int
    average_token_length: float
    numeric_indicator: bool
    punctuation_indicator: bool
    rare_term_proportion: float
    corpus_term_coverage: float


class CorpusFeatures:
    """Document frequency over the unchanged lexical tokenizer; no queries or qrels."""

    def __init__(self, documents: Iterable[Document], rare_fraction: float = 0.01):
        docs = list(documents)
        if not docs or not 0 < rare_fraction <= 1:
            raise ValueError("Require documents and 0 < rare_fraction <= 1")
        self.document_count = len(docs)
        self.rare_df_threshold = max(1, math.ceil(len(docs) * rare_fraction))
        self.document_tokens = {d.id: set(tokenize(d.title + " " + d.text)) for d in docs}
        if len(self.document_tokens) != len(docs):
            raise ValueError("Duplicate document IDs")
        self.df = Counter(token for tokens in self.document_tokens.values() for token in tokens)

    def extract(self, query: str) -> QueryFeatures:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        tokens = tokenize(query)
        unique = set(tokens)
        count = len(unique)
        return QueryFeatures(
            len(tokens),
            count,
            sum(map(len, tokens)) / len(tokens) if tokens else 0.0,
            any(c.isdigit() for c in query),
            any(not c.isalnum() and not c.isspace() for c in query),
            sum(0 < self.df[t] <= self.rare_df_threshold for t in unique) / count if count else 0.0,
            sum(self.df[t] > 0 for t in unique) / count if count else 0.0,
        )


def ranking_diagnostics(bm25: list[SearchResult], dense: list[SearchResult]) -> dict:
    """Optional top-k diagnostics. Not inputs to the cheap live router."""
    result = {}
    for name, ranking in (("bm25", bm25), ("dense", dense)):
        scores = [r.score for r in ranking]
        if not all(math.isfinite(s) for s in scores):
            raise ValueError("Nonfinite diagnostic score")
        result.update(
            {
                f"{name}_score_mean": float(np.mean(scores)) if scores else None,
                f"{name}_score_std": float(np.std(scores)) if scores else None,
                f"{name}_score_max": max(scores) if scores else None,
                f"{name}_score_gap": scores[0] - scores[1] if len(scores) > 1 else None,
            }
        )
    a, b = {r.document_id for r in bm25}, {r.document_id for r in dense}
    result.update(
        {
            "diagnostic_bm25_depth": len(bm25),
            "diagnostic_dense_depth": len(dense),
            "candidate_pool_overlap_topk": len(a & b) / min(len(a), len(b)) if a and b else 0.0,
            "topk_jaccard": len(a & b) / len(a | b) if a | b else 0.0,
            "ranking_position_agreement": sum(
                x.document_id == y.document_id for x, y in zip(bm25, dense)
            )
            / max(len(bm25), len(dense))
            if a or b
            else 0.0,
        }
    )
    return result


def feature_dict(features: QueryFeatures) -> dict:
    return asdict(features)
