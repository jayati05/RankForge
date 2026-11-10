"""Evidence-based, nonexclusive failure indicators; hypotheses are not causes."""

from collections import Counter

from app.evaluation.features import CorpusFeatures
from app.evaluation.metrics import ranking_metrics
from app.retrieval.base import SearchResult

CATEGORIES = (
    "lexical_mismatch",
    "semantic_mismatch",
    "candidate_generation_failure",
    "ranking_failure",
    "reranker_regression",
    "ambiguous_or_underspecified",
    "insufficient_judgments",
    "other_or_unclassified",
)


def classify_failures(
    query: str,
    rankings: dict[str, list[SearchResult]],
    judgments: dict[str, int],
    candidates: list[SearchResult],
    corpus: CorpusFeatures,
    k: int = 10,
) -> list[dict]:
    from app.data.preprocessing import tokenize

    relevant = {did for did, grade in judgments.items() if grade > 0}
    ids = {method: {r.document_id for r in results[:k]} for method, results in rankings.items()}
    pool = {r.document_id for r in candidates}
    indicators = []

    def add(category: str, kind: str, evidence: dict) -> None:
        indicators.append({"category": category, "evidence_type": kind, "evidence": evidence})

    missing = sorted(relevant - pool)
    if missing:
        add(
            "candidate_generation_failure",
            "observed",
            {"missing_judged_positive_ids": missing, "pool_size": len(pool)},
        )
    low = sorted((relevant & pool) - ids["hybrid"])
    if low:
        add(
            "ranking_failure",
            "observed",
            {"judged_positive_in_pool_below_cutoff": low, "cutoff": k},
        )
    before = ranking_metrics([r.document_id for r in rankings["hybrid"][:k]], judgments, k)[
        f"ndcg@{k}"
    ]
    after = ranking_metrics([r.document_id for r in rankings["reranked"][:k]], judgments, k)[
        f"ndcg@{k}"
    ]
    if after < before:
        add(
            "reranker_regression",
            "observed",
            {"hybrid_ndcg": before, "reranked_ndcg": after, "delta": after - before},
        )
    no_overlap = sorted(
        did
        for did in relevant - ids["bm25"]
        if not set(tokenize(query)) & corpus.document_tokens.get(did, set())
    )
    if no_overlap:
        add(
            "lexical_mismatch",
            "inferred_indicator",
            {
                "no_shared_tokens_judged_positive_ids": no_overlap,
                "caveat": "Zero tokenizer overlap and absent from BM25 top-k; not proof of causal failure",
            },
        )
    dense_only_misses = sorted((relevant - ids["dense"]) & ids["bm25"])
    if dense_only_misses:
        add(
            "semantic_mismatch",
            "inferred_indicator",
            {
                "bm25_hits_missing_from_dense_topk": dense_only_misses,
                "caveat": "Differential retrieval is observed; semantic cause is not established",
            },
        )
    features = corpus.extract(query)
    if features.token_count <= 3:
        add(
            "ambiguous_or_underspecified",
            "inferred_indicator",
            {
                "token_count": features.token_count,
                "rule": "At most 3 lexical tokens is a short-query heuristic, not an ambiguity label",
            },
        )
    if len(relevant) <= 1:
        add(
            "insufficient_judgments",
            "inferred_indicator",
            {
                "judgment_count": len(judgments),
                "positive_judgment_count": len(relevant),
                "rule": "At most one positive judgment; evidence sparsity warning, not a claim labels are wrong",
            },
        )
    if not indicators:
        add(
            "other_or_unclassified",
            "observed",
            {"rule": "None of the defined indicators triggered; does not imply failure"},
        )
    return indicators


def query_comparison(
    qid: str,
    query: str,
    rankings: dict[str, list[SearchResult]],
    judgments: dict[str, int],
    candidates: list[SearchResult],
    timings: dict,
    corpus: CorpusFeatures,
    k: int,
) -> tuple[list[dict], dict]:
    relevant = {did for did, grade in judgments.items() if grade > 0}
    metrics = {
        m: ranking_metrics([r.document_id for r in results[:k]], judgments, k)
        for m, results in rankings.items()
    }
    rows = []
    for method, results in rankings.items():
        positions = [r.rank for r in results[:k] if r.document_id in relevant]
        delta = metrics[method][f"ndcg@{k}"] - metrics["bm25"][f"ndcg@{k}"]
        timing = timings[method]
        rows.append(
            {
                "query_id": qid,
                "method": method,
                **metrics[method],
                "first_relevant_rank_at_k": min(positions) if positions else None,
                "relevant_retrieved": len(positions),
                "relevant_missed": len(relevant) - len(positions),
                "judgment_count": len(judgments),
                "positive_judgment_count": len(relevant),
                "judgment_status": "positive_judgments"
                if relevant
                else "no_positive_judgments"
                if judgments
                else "no_judgments",
                "candidate_pool_size": len(candidates)
                if method in ("hybrid", "reranked")
                else None,
                "latency_ms": float(timing["latency_ms"]),
                "rerank_ms": float(timing["rerank_ms"]),
                "retrieval_latency_ms": max(
                    0.0, float(timing["latency_ms"]) - float(timing["rerank_ms"])
                ),
                "ndcg_delta_vs_bm25": delta,
                "effect_vs_bm25": "improved"
                if delta > 0
                else "degraded"
                if delta < 0
                else "unchanged",
            }
        )
    indicators = classify_failures(query, rankings, judgments, candidates, corpus, k)
    analysis = {
        "indicators": indicators,
        "judgments": judgments,
        "rankings": {m: [r.document_id for r in rs] for m, rs in rankings.items()},
        "relevant_positions": {
            m: {did: next((r.rank for r in rs if r.document_id == did), None) for did in relevant}
            for m, rs in rankings.items()
        },
        "pairwise_ndcg_deltas": {
            f"{a}_minus_{b}": metrics[a][f"ndcg@{k}"] - metrics[b][f"ndcg@{k}"]
            for a in rankings
            for b in rankings
            if a != b
        },
        "pairwise_effects": {
            f"{a}_vs_{b}": "improved"
            if metrics[a][f"ndcg@{k}"] > metrics[b][f"ndcg@{k}"]
            else "degraded"
            if metrics[a][f"ndcg@{k}"] < metrics[b][f"ndcg@{k}"]
            else "unchanged"
            for a in rankings
            for b in rankings
            if a != b
        },
        "unjudged_convention": "Gain zero for evaluation; not known irrelevant",
    }
    return rows, analysis


def category_summary(analyses: dict) -> list[dict]:
    counts = Counter(
        indicator["category"] for a in analyses.values() for indicator in a["indicators"]
    )
    return [
        {
            "category": category,
            "query_count": counts[category],
            "query_fraction": counts[category] / len(analyses),
            "nonexclusive": True,
        }
        for category in CATEGORIES
    ]
