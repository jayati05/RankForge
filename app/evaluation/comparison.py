"""Phase 2 comparison. Reuses Phase 1 data selection, BM25 and metric definitions."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from app.data.loaders import load_dataset
from app.data.selection import select_queries
from app.evaluation.benchmark import write_csv
from app.evaluation.metrics import ranking_metrics
from app.retrieval.base import SearchResult, validate_top_k
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.neural import (
    CrossEncoderScorer,
    Encoder,
    PairScorer,
    SentenceEncoder,
    positive_integer,
)
from app.retrieval.reranker import RerankingRetriever

METHODS = ("bm25", "dense", "hybrid", "reranked")
BASELINE = {"k1": 1.5, "b": 0.75, "epsilon": 0.25, "tokenizer": "NFKC-casefold-identifier-regex-v1"}


@dataclass
class ComparisonConfig:
    split: str = "test"
    k: int = 10
    max_queries: int | None = None
    seed: int = 42
    candidate_depth: int = 50
    rrf_constant: float = 60
    rerank_candidates: int = 20
    batch_size: int = 16
    threads: int = 4
    dense_path: str = "models/dense"
    rerank_path: str = "models/reranker"
    cache: str | None = "indexes/scifact-minilm.npz"
    require_cache: bool = False


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def versions() -> dict:
    values = {}
    for name in (
        "numpy",
        "rank-bm25",
        "sentence-transformers",
        "transformers",
        "torch",
        "huggingface-hub",
    ):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "not installed"
    return values


def analyze_query(
    rankings: dict[str, list[SearchResult]], judgments: dict[str, int], candidates: dict, k: int
) -> dict:
    relevant = {did: grade for did, grade in judgments.items() if grade > 0}
    positions = {
        method: {
            did: next((r.rank for r in results if r.document_id == did), None) for did in relevant
        }
        for method, results in rankings.items()
    }
    pool_positions = {
        kind: {
            did: next((r["rank"] for r in results if r["document_id"] == did), None)
            for did in relevant
        }
        for kind, results in candidates.items()
    }
    metrics = {
        method: ranking_metrics([r.document_id for r in results], judgments, k)
        for method, results in rankings.items()
    }
    delta = metrics["reranked"][f"ndcg@{k}"] - metrics["hybrid"][f"ndcg@{k}"]
    return {
        "relevant_judgments": relevant,
        "positions_in_top_k": positions,
        "positions_in_candidate_pool": pool_positions,
        "candidate_presence": {did: pool_positions["hybrid"][did] is not None for did in relevant},
        "metrics": metrics,
        "rerank_ndcg_delta": delta,
        "rerank_effect": "improved" if delta > 0 else "degraded" if delta < 0 else "unchanged",
        "disagree": len({tuple(r.document_id for r in ranking) for ranking in rankings.values()})
        > 1,
        "interpretation": "Positions are observations. Null in top-k means not retrieved at this cutoff; candidate-pool absence cannot be recovered by reranking. No root cause is inferred.",
    }


def summaries(rows: list[dict], k: int) -> list[dict]:
    result = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        latency = [row["latency_ms"] for row in selected]
        result.append(
            {
                "method": method,
                "query_count": len(selected),
                **{
                    f"{metric}@{k}": mean(row[f"{metric}@{k}"] for row in selected)
                    for metric in ("recall", "precision", "mrr", "ndcg")
                },
                "mean_latency_ms": mean(latency),
                "p50_latency_ms": float(np.percentile(latency, 50)) if len(latency) >= 20 else None,
                "p95_latency_ms": float(np.percentile(latency, 95)) if len(latency) >= 20 else None,
                "mean_rerank_ms": mean(row["rerank_ms"] for row in selected),
                "mean_rerank_candidates": mean(row["candidate_count"] for row in selected),
            }
        )
    return result


def run_comparison(
    dataset_path: Path,
    output: Path,
    config: ComparisonConfig,
    encoder: Encoder | None = None,
    scorer: PairScorer | None = None,
) -> Path:
    validate_top_k(config.k)
    for name in ("candidate_depth", "rerank_candidates", "batch_size", "threads"):
        positive_integer(getattr(config, name), name)
    if config.rerank_candidates < config.k or config.candidate_depth < config.rerank_candidates:
        raise ValueError("Require k <= rerank_candidates <= candidate_depth")
    run_id = (
        "phase2-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    run_dir = Path(output) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "status": "initializing",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "baseline_configuration": BASELINE,
        "versions": versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "device": "cpu",
        "injected_models": encoder is not None or scorer is not None,
        "timing": "One authored-text warmup search per complete pipeline. Independent pipeline execution with deterministic rotated order per query. Timed latency includes dependencies, excludes model loading/index setup/metrics/I/O. No query embedding or result reuse between pipelines.",
        "percentiles": "NumPy linear interpolation; omitted for fewer than 20 queries; single pass, no confidence intervals.",
        "evaluation": "Unchanged Phase 1 metrics; macro mean, grade>0 binary relevance, exponential nDCG, precision denominator k. Unjudged gain=0 does not assert irrelevance.",
        "code_sha256": {
            p.relative_to(Path(__file__).resolve().parents[2]).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        },
    }
    rows: list[dict] = []
    rankings: dict[str, dict] = {method: {} for method in METHODS}
    pools, failures = {}, {}

    def checkpoint() -> None:
        if rows:
            write_csv(run_dir / "per_query_results.csv", rows)
        write_json(run_dir / "rankings.json", rankings)
        write_json(run_dir / "candidate_pools.json", pools)
        write_json(run_dir / "failure_cases.json", failures)
        metadata["completed_queries_by_method"] = {
            method: len(values) for method, values in rankings.items()
        }
        write_json(run_dir / "run_metadata.json", metadata)

    checkpoint()
    try:
        full_data = load_dataset(dataset_path, config.split)
        data = select_queries(full_data, config.max_queries, config.seed)
        if config.rerank_candidates >= len(data.documents):
            raise ValueError(
                "Rerank candidate pool must be smaller than corpus; reduce it for smoke data"
            )
        sources = ["corpus.jsonl", "queries.jsonl", f"qrels/{config.split}.tsv"]
        if (Path(dataset_path) / "metadata.json").exists():
            sources.append("metadata.json")
        metadata.update(
            {
                "dataset": data.metadata,
                "dataset_path": str(Path(dataset_path).resolve()),
                "document_count": len(data.documents),
                "query_count": len(data.queries),
                "available_query_count": len(full_data.queries),
                "query_ids": list(data.queries),
                "judgment_count": sum(map(len, data.qrels.values())),
                "source_sha256": {
                    name: hashlib.sha256((Path(dataset_path) / name).read_bytes()).hexdigest()
                    for name in sources
                },
            }
        )
        load_times = {}
        start = perf_counter()
        if encoder is None:
            encoder = SentenceEncoder(config.dense_path, config.threads, config.seed)
        load_times["dense"] = perf_counter() - start
        checkpoint()
        start = perf_counter()
        if scorer is None:
            scorer = CrossEncoderScorer(config.rerank_path, config.threads, config.seed)
        load_times["reranker"] = perf_counter() - start
        metadata["model_load_seconds"] = load_times
        metadata["models"] = {"dense": encoder.identity, "reranker": scorer.identity}
        start = perf_counter()
        bm25 = BM25Retriever(data.documents.values())
        bm25_seconds = perf_counter() - start
        start = perf_counter()
        dense = DenseRetriever(
            data.documents.values(),
            encoder,
            config.batch_size,
            Path(config.cache) if config.cache else None,
            config.require_cache,
        )
        dense_seconds = perf_counter() - start
        hybrid = HybridRetriever(bm25, dense, config.candidate_depth, config.rrf_constant)
        reranked = RerankingRetriever(
            hybrid, data.documents.values(), scorer, config.rerank_candidates, config.batch_size
        )
        systems = dict(zip(METHODS, (bm25, dense, hybrid, reranked)))
        metadata["index_setup_seconds"] = {
            "bm25": bm25_seconds,
            "dense": dense_seconds,
            "hybrid": bm25_seconds + dense_seconds,
            "reranked": bm25_seconds + dense_seconds,
        }
        metadata["dense_cache"] = {
            "path": config.cache,
            "status": dense.cache_status,
            "original_build_seconds": dense.original_build_seconds,
            "identity": dense.metadata,
        }
        warmup_times = {}
        for method, system in systems.items():
            start = perf_counter()
            system.search("RankForge warmup scientific evidence", config.k)
            warmup_times[method] = perf_counter() - start
        metadata.update({"warmup_seconds": warmup_times, "status": "running"})
        checkpoint()
        for index, (qid, query) in enumerate(data.queries.items()):
            method_order = METHODS[index % 4 :] + METHODS[: index % 4]
            query_results = {}
            for method in method_order:
                start = perf_counter()
                trace = None
                if method == "reranked":
                    trace = reranked.search_with_trace(query.text, config.k)
                    results = trace.results
                else:
                    results = systems[method].search(query.text, config.k)
                latency_ms = (perf_counter() - start) * 1000
                query_results[method] = results
                rankings[method][qid] = [asdict(r) for r in results]
                rows.append(
                    {
                        "method": method,
                        "query_id": qid,
                        "latency_ms": latency_ms,
                        "candidate_count": len(trace.candidates) if trace else 0,
                        "rerank_ms": trace.rerank_ms if trace else 0.0,
                        **ranking_metrics(
                            [r.document_id for r in results], data.qrels[qid], config.k
                        ),
                    }
                )
                if trace:
                    pools[qid] = {
                        "hybrid": [asdict(r) for r in trace.candidates],
                        "reranked": [asdict(r) for r in trace.reranked_candidates],
                    }
            analysis = analyze_query(query_results, data.qrels[qid], pools[qid], config.k)
            if analysis["disagree"]:
                failures[qid] = analysis
            if (index + 1) % 10 == 0:
                checkpoint()
                print(f"{run_id}: {index + 1}/{len(data.queries)} queries complete", flush=True)
        summary = summaries(rows, config.k)
        write_csv(run_dir / "benchmark_summary.csv", summary)
        write_json(run_dir / "benchmark_summary.json", summary)
        metadata.update(
            {"status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        )
        checkpoint()
        replay_comparison(run_dir, dataset_path)
        return run_dir
    except BaseException as exc:
        metadata.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        checkpoint()
        print(f"Incomplete comparison saved at {run_dir}", file=sys.stderr, flush=True)
        raise


def replay_comparison(run_dir: Path, dataset_path: Path) -> int:
    """Verify original input hashes, identical query sets, rankings, qrels joins and saved metrics."""
    run_dir, dataset_path = Path(run_dir), Path(dataset_path)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("Cannot replay an incomplete comparison as complete")
    config = ComparisonConfig(**metadata["config"])
    data = select_queries(load_dataset(dataset_path, config.split), config.max_queries, config.seed)
    if list(data.queries) != metadata["query_ids"]:
        raise ValueError("Query selection changed")
    for name, digest in metadata["source_sha256"].items():
        if hashlib.sha256((dataset_path / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Dataset file changed")
    rankings = json.loads((run_dir / "rankings.json").read_text(encoding="utf-8"))
    if set(rankings) != set(METHODS) or any(set(rankings[m]) != set(data.queries) for m in METHODS):
        raise ValueError("Methods were not evaluated on identical queries")
    pools = json.loads((run_dir / "candidate_pools.json").read_text(encoding="utf-8"))
    with (run_dir / "per_query_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(data.queries) * len(METHODS) or len(
        {(r["method"], r["query_id"]) for r in rows}
    ) != len(rows):
        raise ValueError("Missing or duplicate per-query metric rows")
    if {(r["method"], r["query_id"]) for r in rows} != {
        (method, qid) for method in METHODS for qid in data.queries
    }:
        raise ValueError("Per-query rows have inconsistent method/query IDs")
    if set(pools) != set(data.queries):
        raise ValueError("Missing or extra candidate-pool queries")
    for row in rows:
        method, qid = row["method"], row["query_id"]
        results = rankings[method][qid]
        ids = [r["document_id"] for r in results]
        if len(ids) > config.k or any(did not in data.documents for did in ids):
            raise ValueError("Invalid ranking documents or cutoff")
        if [r["rank"] for r in results] != list(range(1, len(results) + 1)) or any(
            not math.isfinite(r["score"]) for r in results
        ):
            raise ValueError("Invalid ranking ranks or scores")
        expected = ranking_metrics(ids, data.qrels[qid], config.k)
        if any(
            not math.isclose(float(row[key]), value, abs_tol=1e-12)
            for key, value in expected.items()
        ):
            raise ValueError("Saved metric does not match ranking/qrels")
        for key in ("latency_ms", "rerank_ms", "candidate_count", *expected):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Invalid metric or timing value")
        if method == "reranked":
            before, after = pools[qid]["hybrid"], pools[qid]["reranked"]
            before_ids, after_ids = (
                [r["document_id"] for r in before],
                [r["document_id"] for r in after],
            )
            if (
                len(before_ids) > config.rerank_candidates
                or len(before_ids) != len(set(before_ids))
                or sorted(before_ids) != sorted(after_ids)
                or ids != after_ids[: config.k]
                or int(row["candidate_count"]) != len(before_ids)
            ):
                raise ValueError("Reranker candidate provenance mismatch")
    for qid in data.queries:
        for kind, results in pools[qid].items():
            if kind not in ("hybrid", "reranked") or [r["rank"] for r in results] != list(
                range(1, len(results) + 1)
            ):
                raise ValueError("Invalid candidate-pool ranks")
            if any(
                r["document_id"] not in data.documents or not math.isfinite(r["score"])
                for r in results
            ):
                raise ValueError("Invalid candidate-pool IDs/scores")
        if rankings["hybrid"][qid] != pools[qid]["hybrid"][: config.k]:
            raise ValueError("Hybrid top-k does not match reranker candidate prefix")
    saved_failures = json.loads((run_dir / "failure_cases.json").read_text(encoding="utf-8"))
    expected_failures = {}
    for qid in data.queries:
        analysis = analyze_query(
            {
                method: [SearchResult(**entry) for entry in rankings[method][qid]]
                for method in METHODS
            },
            data.qrels[qid],
            pools[qid],
            config.k,
        )
        if analysis["disagree"]:
            expected_failures[qid] = analysis
    if saved_failures != expected_failures:
        raise ValueError("Disagreement analysis does not match saved rankings/judgments")
    saved_summary = json.loads((run_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    expected_summary = summaries(rows, config.k)
    for saved, expected in zip(saved_summary, expected_summary, strict=True):
        for key, value in expected.items():
            if isinstance(value, (int, float)):
                if not math.isclose(saved[key], value, abs_tol=1e-12):
                    raise ValueError("Summary differs from per-query rows")
            elif saved[key] != value:
                raise ValueError("Invalid summary value")
    with (run_dir / "benchmark_summary.csv").open(encoding="utf-8", newline="") as stream:
        csv_summary = list(csv.DictReader(stream))
    if len(csv_summary) != len(expected_summary):
        raise ValueError("Summary CSV has the wrong number of methods")
    for saved, expected in zip(csv_summary, expected_summary, strict=True):
        for key, value in expected.items():
            if isinstance(value, (int, float)):
                if not math.isclose(float(saved[key]), value, abs_tol=1e-12):
                    raise ValueError("Summary CSV differs from per-query rows")
            elif saved[key] != ("" if value is None else value):
                raise ValueError("Invalid summary CSV value")
    return len(rows)


def main() -> None:
    from app.tracking.mlflow import add_arguments, track_arguments

    parser = argparse.ArgumentParser(
        description="Compare BM25, dense, RRF and cross-encoder reranking"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/public/scifact"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/phase2"))
    parser.add_argument(
        "--replay", type=Path, help="Validate a saved completed run without loading models"
    )
    for name, default, kind in [
        ("split", "test", str),
        ("k", 10, int),
        ("max-queries", None, int),
        ("seed", 42, int),
        ("candidate-depth", 50, int),
        ("rrf-constant", 60, float),
        ("rerank-candidates", 20, int),
        ("batch-size", 16, int),
        ("threads", 4, int),
        ("dense-path", "models/dense", str),
        ("rerank-path", "models/reranker", str),
        ("cache", "indexes/scifact-minilm.npz", str),
    ]:
        parser.add_argument("--" + name, default=default, type=kind)
    parser.add_argument("--require-cache", action="store_true")
    add_arguments(parser)
    args = parser.parse_args()
    try:
        if args.replay:
            print(f"Verified {replay_comparison(args.replay, args.dataset)} per-query rows")
            track_arguments(args, args.replay, historical=True)
        else:
            config = ComparisonConfig(
                **{name: getattr(args, name) for name in ComparisonConfig.__dataclass_fields__}
            )
            source = run_comparison(args.dataset, args.output, config)
            print(source)
            track_arguments(args, source)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
