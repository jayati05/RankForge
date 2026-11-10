"""Split-aware policy selection, fresh live evaluation and paired uncertainty."""

import hashlib
import json
from dataclasses import asdict
from statistics import mean
from time import perf_counter

import numpy as np

from app.evaluation.comparison import METHODS
from app.evaluation.features import QueryFeatures
from app.evaluation.metrics import ranking_metrics
from app.evaluation.splits import validate_partition
from app.retrieval.router import QueryAwareRouter, RoutingPolicy


UTILITY_PENALTY_PER_SECOND = 0.1


def select_policy(features: dict[str, QueryFeatures], outcomes: dict, split: dict) -> dict:
    """Reject held-out labels; shortlist on development, select on validation only."""
    validate_partition(split, list(features))
    permitted = set(split["development"]) | set(split["validation"])
    if set(outcomes) != permitted:
        raise ValueError(
            "Tuning outcomes must contain development/validation only; held-out labels prohibited"
        )
    candidates = [
        RoutingPolicy(rare, length) for rare in (0.1, 0.25, 0.5, 1.01) for length in (0, 8, 16)
    ]

    def assess(policy: RoutingPolicy, ids: list[str]) -> dict:
        routes = [policy.decide(features[q])[0] for q in ids]
        quality = mean(outcomes[q][m]["ndcg"] for q, m in zip(ids, routes))
        latency = mean(outcomes[q][m]["latency_ms"] for q, m in zip(ids, routes))
        rate = routes.count("reranked") / len(routes)
        return {
            "policy": asdict(policy),
            "ndcg": quality,
            "estimated_latency_ms": latency,
            "reranker_rate": rate,
            "utility": quality - UTILITY_PENALTY_PER_SECOND * latency / 1000,
        }

    development = [assess(p, split["development"]) for p in candidates]
    ordered = sorted(range(len(candidates)), key=lambda i: (-development[i]["utility"], i))
    shortlist = [i for i in ordered if development[i]["reranker_rate"] <= 0.5][:6]
    fallback = candidates.index(RoutingPolicy(1.01, 0))
    shortlist = sorted(set(shortlist + [fallback]))
    validation = [
        {"candidate_index": i, **assess(candidates[i], split["validation"])} for i in shortlist
    ]
    eligible = [row for row in validation if row["reranker_rate"] <= 0.5]
    winner = sorted(
        eligible,
        key=lambda row: (-row["utility"], row["estimated_latency_ms"], row["candidate_index"]),
    )[0]
    return {
        "policy": winner["policy"],
        "development": development,
        "validation": validation,
        "selected_candidate_index": winner["candidate_index"],
        "label_query_ids": sorted(permitted),
        "objective": "mean nDCG@k - 0.1 * mean source latency in seconds; <=50% reranker rate on development/validation",
        "latency_source": "Historical Phase 2 same-query measurements; estimates for selection, not new live timing",
        "no_held_out_labels": True,
    }


def freeze_digest(policy_record: dict) -> str:
    return hashlib.sha256(
        json.dumps(policy_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def paired_bootstrap(a: list[float], b: list[float], seed: int = 42, samples: int = 2000) -> dict:
    if len(a) != len(b) or not a or samples <= 0:
        raise ValueError("Require equal nonempty paired observations and positive samples")
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite bootstrap observations")
    rng = np.random.default_rng(seed)
    estimates = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
    return {
        "mean_difference": float(delta.mean()),
        "ci95_lower": float(np.percentile(estimates, 2.5)),
        "ci95_upper": float(np.percentile(estimates, 97.5)),
        "samples": samples,
        "seed": seed,
        "unit": "paired query bootstrap, percentile interval; exploratory, not corrected for comparisons",
    }


def evaluate_live(
    data,
    query_ids: list[str],
    systems: dict,
    corpus,
    policy: RoutingPolicy,
    k: int,
    checkpoint=None,
) -> tuple[list[dict], list[dict], dict]:
    """The router is called before qrels are read for that result; every baseline runs independently."""
    router = QueryAwareRouter(corpus, systems, policy)
    all_systems = {**systems, "router": router}
    names = (*METHODS, "router")
    for name in names:
        all_systems[name].search("RankForge warmup scientific evidence", k)
    rows, decisions, saved = [], [], {name: {} for name in names}
    for i, qid in enumerate(query_ids):
        for name in names[i % 5 :] + names[: i % 5]:
            query = data.queries[qid].text
            started = perf_counter()
            trace = None
            if name == "router":
                trace = router.search_with_trace(query, k)
                results = trace.results
            elif name == "reranked":
                trace = systems[name].search_with_trace(query, k)
                results = trace.results
            else:
                results = all_systems[name].search(query, k)
            latency = (perf_counter() - started) * 1000
            saved[name][qid] = [asdict(r) for r in results]
            metrics = ranking_metrics([r.document_id for r in results], data.qrels.get(qid, {}), k)
            if name == "router":
                decisions.append(
                    {
                        "query_id": qid,
                        "strategy": trace.strategy,
                        "explanation": trace.explanation,
                        "features": json.dumps(trace.features, sort_keys=True),
                        "reranker_invoked": trace.reranker_invoked,
                        "candidate_count": trace.candidate_count,
                        "latency_ms": latency,
                        "routing_overhead_ms": trace.routing_overhead_ms,
                        "rerank_ms": trace.rerank_ms,
                    }
                )
                invoked, count, rerank_ms = (
                    trace.reranker_invoked,
                    trace.candidate_count,
                    trace.rerank_ms,
                )
            elif name == "reranked":
                invoked, count, rerank_ms = True, len(trace.candidates), trace.rerank_ms
            else:
                invoked, count, rerank_ms = False, 0, 0.0
            rows.append(
                {
                    "query_id": qid,
                    "method": name,
                    **metrics,
                    "latency_ms": latency,
                    "reranker_invoked": invoked,
                    "candidate_count": count,
                    "rerank_ms": rerank_ms,
                }
            )
        if checkpoint:
            checkpoint(rows, decisions, saved)
        if (i + 1) % 10 == 0:
            print(f"Held-out live evaluation: {i + 1}/{len(query_ids)} queries", flush=True)
    return rows, decisions, saved


def evaluate_summary(rows: list[dict], decisions: list[dict], k: int, seed: int = 42) -> dict:
    names = (*METHODS, "router")
    by_method = {name: {r["query_id"]: r for r in rows if r["method"] == name} for name in names}
    ids = sorted(by_method["router"])
    if not ids or any(set(values) != set(ids) for values in by_method.values()):
        raise ValueError("All baselines/router must cover identical queries")
    summary = []
    for name in names:
        values = [by_method[name][q] for q in ids]
        latency = [r["latency_ms"] for r in values]
        summary.append(
            {
                "method": name,
                "query_count": len(values),
                **{
                    f"{m}@{k}": mean(r[f"{m}@{k}"] for r in values)
                    for m in ("recall", "precision", "mrr", "ndcg")
                },
                "mean_latency_ms": mean(latency),
                "p50_latency_ms": float(np.percentile(latency, 50)),
                "p95_latency_ms": float(np.percentile(latency, 95)),
                "reranker_invocation_rate": mean(bool(r["reranker_invoked"]) for r in values),
            }
        )
    baseline_latency = next(r["mean_latency_ms"] for r in summary if r["method"] == "reranked")
    for row in summary:
        row["latency_savings_vs_reranked_fraction"] = (
            1 - row["mean_latency_ms"] / baseline_latency if baseline_latency else None
        )
    paired = {}
    for baseline in METHODS:
        paired[baseline] = {
            metric: paired_bootstrap(
                [by_method["router"][q][metric] for q in ids],
                [by_method[baseline][q][metric] for q in ids],
                seed,
            )
            for metric in (f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}", "latency_ms")
        }
    distribution = {
        name: sum(d["strategy"] == name for d in decisions) / len(decisions)
        for name in ("bm25", "hybrid", "reranked")
    }
    return {
        "summary": summary,
        "paired_router_minus_baseline": paired,
        "route_fractions": distribution,
        "query_ids": ids,
        "limitations": f"{len(ids)}-query internal exploratory slice of an exposed benchmark. Single warmed timing per system/query; query bootstrap does not capture runtime noise or remove prior exposure bias.",
    }
