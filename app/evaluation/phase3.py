"""Ranking Failure Lab and exploratory routing experiment, preserving prior runs."""

import argparse
import csv
import hashlib
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.data.loaders import load_dataset
from app.evaluation.comparison import METHODS, replay_comparison, versions, write_json
from app.evaluation.benchmark import write_csv
from app.evaluation.failure_lab import category_summary, query_comparison
from app.evaluation.features import CorpusFeatures, feature_dict, ranking_diagnostics
from app.evaluation.metrics import ranking_metrics
from app.evaluation.routing_experiment import (
    evaluate_live,
    evaluate_summary,
    freeze_digest,
    select_policy,
)
from app.evaluation.splits import exploratory_split, validate_partition
from app.retrieval.base import SearchResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.neural import CrossEncoderScorer, SentenceEncoder
from app.retrieval.reranker import RerankingRetriever
from app.retrieval.router import RoutingPolicy

REQUIRED = (
    "failure_cases.csv",
    "per_query_comparison.csv",
    "query_features.csv",
    "failure_summary.csv",
    "system_comparison.csv",
    "routing_decisions.csv",
    "failure_analysis.json",
    "routing_evaluation.json",
    "split_metadata.json",
    "frozen_policy.json",
    "tuning_audit.json",
    "routing_per_query.csv",
    "routing_rankings.json",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def hashes(root: Path, names) -> dict:
    return {str(name): hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def build_systems(data, config: dict) -> tuple[dict, dict]:
    timings = {}
    start = perf_counter()
    encoder = SentenceEncoder(config["dense_path"], config["threads"], config["seed"])
    timings["dense_model_load_seconds"] = perf_counter() - start
    start = perf_counter()
    scorer = CrossEncoderScorer(config["rerank_path"], config["threads"], config["seed"])
    timings["reranker_model_load_seconds"] = perf_counter() - start
    start = perf_counter()
    bm25 = BM25Retriever(data.documents.values())
    dense = DenseRetriever(
        data.documents.values(), encoder, config["batch_size"], Path(config["cache"]), True
    )
    hybrid = HybridRetriever(bm25, dense, config["candidate_depth"], config["rrf_constant"])
    reranked = RerankingRetriever(
        hybrid, data.documents.values(), scorer, config["rerank_candidates"], config["batch_size"]
    )
    timings.update(
        {
            "shared_index_setup_seconds": perf_counter() - start,
            "dense_cache_status": dense.cache_status,
            "models": {"dense": encoder.identity, "reranker": scorer.identity},
        }
    )
    return dict(zip(METHODS, (bm25, dense, hybrid, reranked))), timings


def run_phase3(
    dataset_path: Path,
    source: Path,
    output: Path,
    seed: int = 42,
    systems_override: dict | None = None,
    plots: bool = True,
) -> Path:
    dataset_path, source = Path(dataset_path), Path(source)
    replay_comparison(source, dataset_path)
    source_meta = read_json(source / "run_metadata.json")
    config = source_meta["config"]
    k = config["k"]
    data = load_dataset(dataset_path, config["split"])
    ids = source_meta["query_ids"]
    split = exploratory_split(ids, seed)
    validate_partition(split, ids)
    run_id = (
        "phase3-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    run_dir = Path(output) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "status": "initializing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_phase2": str(source.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "seed": seed,
        "source_run_id": source_meta["run_id"],
        "pipeline_config": config,
        "source_artifact_hashes": hashes(
            source,
            ("run_metadata.json", "rankings.json", "candidate_pools.json", "per_query_results.csv"),
        ),
        "versions": versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "code_sha256": {
            p.relative_to(Path(__file__).resolve().parents[2]).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        },
        "limitation": split["limitation"],
        "events": [],
        "injected_systems": systems_override is not None,
        "timing": "Fresh independent CPU searches, rotated five-system order, one authored-query warmup per system; live router latency includes feature extraction and decision. Loading/index setup excluded. Historical failure-analysis timings remain Phase 2 observations.",
        "zero_judgment_policy": "Retain every source query; empty/missing judgments yield zero metrics per existing definitions and explicit insufficient-judgment evidence. Source benchmark universe is persisted Phase 2 IDs.",
    }

    def event(name: str) -> None:
        metadata["events"].append(
            {"name": name, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        )
        write_json(run_dir / "run_metadata.json", metadata)

    write_json(run_dir / "split_metadata.json", split)
    event("split_persisted")
    try:
        corpus = CorpusFeatures(data.documents.values())
        saved_rankings = read_json(source / "rankings.json")
        source_pools = read_json(source / "candidate_pools.json")
        rankings = {
            q: {m: [SearchResult(**r) for r in saved_rankings[m][q]] for m in METHODS} for q in ids
        }
        source_rows = {
            (r["query_id"], r["method"]): r for r in read_csv(source / "per_query_results.csv")
        }
        features, feature_rows = {}, []
        for q in ids:
            start = perf_counter()
            features[q] = corpus.extract(data.queries[q].text)
            extraction_ms = (perf_counter() - start) * 1000
            feature_rows.append(
                {
                    "query_id": q,
                    **feature_dict(features[q]),
                    "feature_extraction_ms": extraction_ms,
                    **ranking_diagnostics(rankings[q]["bm25"], rankings[q]["dense"]),
                }
            )
        write_csv(run_dir / "query_features.csv", feature_rows)
        permitted = split["development"] + split["validation"]
        outcomes = {
            q: {
                m: {
                    "ndcg": float(source_rows[q, m][f"ndcg@{k}"]),
                    "latency_ms": float(source_rows[q, m]["latency_ms"]),
                }
                for m in METHODS
            }
            for q in permitted
        }
        tuning = select_policy(features, outcomes, split)
        write_json(run_dir / "tuning_audit.json", tuning)
        policy_record = {
            "policy": tuning["policy"],
            "source_run_id": source_meta["run_id"],
            "split_seed": seed,
            "tuning_label_query_ids": tuning["label_query_ids"],
            "selected_on": "validation",
            "objective": tuning["objective"],
            "version": 1,
        }
        write_json(run_dir / "frozen_policy.json", policy_record)
        metadata["frozen_policy_sha256"] = freeze_digest(policy_record)
        event("policy_frozen_before_held_out_evaluation")
        policy = RoutingPolicy(**tuning["policy"])
        if systems_override is None:
            systems, setup = build_systems(data, config)
            metadata["live_setup"] = setup
        else:
            systems = systems_override
        event("held_out_evaluation_started")

        def checkpoint(rows, decisions, live_rankings):
            write_csv(run_dir / "routing_per_query.csv", rows)
            write_csv(run_dir / "routing_decisions.csv", decisions)
            write_json(run_dir / "routing_rankings.json", live_rankings)
            metadata["completed_held_out_queries"] = len(decisions)
            write_json(run_dir / "run_metadata.json", metadata)

        live_rows, decisions, live_rankings = evaluate_live(
            data, split["held_out"], systems, corpus, policy, k, checkpoint
        )
        checkpoint(live_rows, decisions, live_rankings)
        if (
            freeze_digest(read_json(run_dir / "frozen_policy.json"))
            != metadata["frozen_policy_sha256"]
        ):
            raise ValueError("Policy changed during held-out evaluation")
        evaluation = evaluate_summary(live_rows, decisions, k, seed)
        evaluation["frozen_policy_sha256"] = metadata["frozen_policy_sha256"]
        write_json(run_dir / "routing_evaluation.json", evaluation)
        write_csv(run_dir / "system_comparison.csv", evaluation["summary"])
        event("held_out_evaluation_completed_without_policy_update")
        analyses, comparison_rows, failure_rows = {}, [], []
        for q in ids:
            comparisons, analysis = query_comparison(
                q,
                data.queries[q].text,
                rankings[q],
                data.qrels.get(q, {}),
                [SearchResult(**r) for r in source_pools[q]["hybrid"]],
                {m: source_rows[q, m] for m in METHODS},
                corpus,
                k,
            )
            analyses[q] = analysis
            comparison_rows.extend(comparisons)
            failure_rows.extend(
                {
                    "query_id": q,
                    "category": indicator["category"],
                    "evidence_type": indicator["evidence_type"],
                    "evidence": json.dumps(indicator["evidence"], sort_keys=True),
                }
                for indicator in analysis["indicators"]
            )
        write_csv(run_dir / "per_query_comparison.csv", comparison_rows)
        write_csv(run_dir / "failure_cases.csv", failure_rows)
        write_csv(run_dir / "failure_summary.csv", category_summary(analyses))
        write_json(run_dir / "failure_analysis.json", analyses)
        if plots:
            from app.evaluation.phase3_plots import generate_plots

            generate_plots(run_dir)
        metadata.update(
            {
                "status": "complete",
                "required_artifacts": list(REQUIRED),
                "plots_enabled": plots,
                "artifact_sha256": hashes(run_dir, REQUIRED),
                "analysis_query_count": len(ids),
                "held_out_query_count": len(split["held_out"]),
            }
        )
        if plots:
            metadata["plot_sha256"] = hashes(
                run_dir,
                [
                    p.relative_to(run_dir).as_posix()
                    for p in sorted((run_dir / "plots").glob("*.png"))
                ],
            )
        event("complete")
        replay_phase3(run_dir, dataset_path)
        return run_dir
    except BaseException as exc:
        metadata.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        event("failed")
        print(f"Incomplete Phase 3 run retained at {run_dir}", file=sys.stderr)
        raise


def replay_phase3(run_dir: Path, dataset_path: Path) -> int:
    run_dir = Path(run_dir)
    metadata = read_json(run_dir / "run_metadata.json")
    if metadata["status"] != "complete":
        raise ValueError("Cannot replay incomplete Phase 3 run")
    for key in ("artifact_sha256", "plot_sha256"):
        for name, expected in metadata.get(key, {}).items():
            if hashes(run_dir, [name])[name] != expected:
                raise ValueError(f"Artifact changed: {name}")
    source = Path(metadata["source_phase2"])
    replay_comparison(source, dataset_path)
    if hashes(source, metadata["source_artifact_hashes"]) != metadata["source_artifact_hashes"]:
        raise ValueError("Phase 2 source artifacts changed")
    split = read_json(run_dir / "split_metadata.json")
    source_metadata = read_json(source / "run_metadata.json")
    if split != exploratory_split(source_metadata["query_ids"], metadata["seed"]):
        raise ValueError("Split selection changed")
    validate_partition(split, source_metadata["query_ids"])
    policy_record = read_json(run_dir / "frozen_policy.json")
    if freeze_digest(policy_record) != metadata["frozen_policy_sha256"]:
        raise ValueError("Frozen policy changed")
    if set(policy_record["tuning_label_query_ids"]) != set(
        split["development"] + split["validation"]
    ):
        raise ValueError("Policy tuning includes forbidden IDs")
    event_names = [e["name"] for e in metadata["events"]]
    if event_names.index("policy_frozen_before_held_out_evaluation") >= event_names.index(
        "held_out_evaluation_started"
    ):
        raise ValueError("Policy was not frozen before evaluation")
    data = load_dataset(dataset_path, source_metadata["config"]["split"])
    corpus = CorpusFeatures(data.documents.values())
    feature_values = {q: corpus.extract(data.queries[q].text) for q in source_metadata["query_ids"]}
    source_rows = {
        (r["query_id"], r["method"]): r for r in read_csv(source / "per_query_results.csv")
    }
    permitted = split["development"] + split["validation"]
    k = source_metadata["config"]["k"]
    outcomes = {
        q: {
            m: {
                "ndcg": float(source_rows[q, m][f"ndcg@{k}"]),
                "latency_ms": float(source_rows[q, m]["latency_ms"]),
            }
            for m in METHODS
        }
        for q in permitted
    }
    if select_policy(feature_values, outcomes, split) != read_json(run_dir / "tuning_audit.json"):
        raise ValueError("Policy selection audit cannot be reproduced")
    policy = RoutingPolicy(**policy_record["policy"])
    rows = read_csv(run_dir / "routing_per_query.csv")
    decisions = read_csv(run_dir / "routing_decisions.csv")
    rankings = read_json(run_dir / "routing_rankings.json")
    if set(rankings) != {*METHODS, "router"} or any(
        set(values) != set(split["held_out"]) for values in rankings.values()
    ):
        raise ValueError("Live systems have different held-out queries")
    if len(rows) != 5 * len(split["held_out"]) or len(
        {(r["method"], r["query_id"]) for r in rows}
    ) != len(rows):
        raise ValueError("Missing or duplicate live rows")
    if len(decisions) != len(split["held_out"]) or {d["query_id"] for d in decisions} != set(
        split["held_out"]
    ):
        raise ValueError("Invalid decision query IDs")
    for decision in decisions:
        q = decision["query_id"]
        strategy, explanation = policy.decide(feature_values[q])
        if (
            decision["strategy"] != strategy
            or decision["explanation"] != explanation
            or json.loads(decision["features"]) != feature_dict(feature_values[q])
        ):
            raise ValueError("Decision depends on something other than inference-time features")
        if [r["document_id"] for r in rankings["router"][q]] != [
            r["document_id"] for r in rankings[strategy][q]
        ]:
            raise ValueError("Router output does not match selected independent baseline")
    for row in rows:
        expected = ranking_metrics(
            [r["document_id"] for r in rankings[row["method"]][row["query_id"]]],
            data.qrels.get(row["query_id"], {}),
            k,
        )
        if any(abs(float(row[key]) - value) > 1e-12 for key, value in expected.items()):
            raise ValueError("Live ranking metrics do not replay")
        for key in (*expected, "latency_ms", "candidate_count", "rerank_ms"):
            row[key] = float(row[key])
        row["reranker_invoked"] = row["reranker_invoked"] == "True"
    computed = evaluate_summary(rows, decisions, k, metadata["seed"])
    computed["frozen_policy_sha256"] = metadata["frozen_policy_sha256"]
    if computed != read_json(run_dir / "routing_evaluation.json"):
        raise ValueError("Routing evaluation cannot be reproduced")
    return len(rows)


def main() -> None:
    from app.tracking.mlflow import add_arguments, track_arguments

    parser = argparse.ArgumentParser(
        description="Ranking Failure Lab and exploratory query-aware routing"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/public/scifact"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104"),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/results/phase3"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay", type=Path)
    add_arguments(parser)
    args = parser.parse_args()
    try:
        if args.replay:
            print(f"Verified {replay_phase3(args.replay, args.dataset)} live baseline/router rows")
            track_arguments(args, args.replay, historical=True)
        else:
            source = run_phase3(args.dataset, args.source, args.output, args.seed)
            print(source)
            track_arguments(args, source)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Phase 3 failed: {exc}\n")


if __name__ == "__main__":
    main()
