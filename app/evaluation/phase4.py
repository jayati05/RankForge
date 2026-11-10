"""External NFCorpus validation of the immutable Phase 3 policy; no tuning API."""

import argparse
import json
import math
import platform
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.data.download import file_hash
from app.data.loaders import load_dataset
from app.evaluation.benchmark import write_csv
from app.evaluation.comparison import METHODS, versions, write_json
from app.evaluation.failure_lab import category_summary, query_comparison
from app.evaluation.features import CorpusFeatures, feature_dict
from app.evaluation.metrics import ranking_metrics
from app.evaluation.phase3 import hashes, read_csv, read_json
from app.evaluation.routing_experiment import evaluate_live, evaluate_summary, paired_bootstrap
from app.evaluation.transfer_policy import load_protocol
from app.retrieval.base import SearchResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.neural import CrossEncoderScorer, SentenceEncoder
from app.retrieval.reranker import RerankingRetriever

REQUIRED = (
    "system_comparison.csv",
    "per_query_comparison.csv",
    "failure_cases.csv",
    "failure_summary.csv",
    "routing_decisions.csv",
    "experiment_metadata.json",
    "dataset_provenance.json",
    "generalization_analysis.json",
    "phase4_report.json",
)
PAYLOAD = (
    "system_comparison.csv",
    "per_query_comparison.csv",
    "failure_cases.csv",
    "failure_summary.csv",
    "routing_decisions.csv",
    "dataset_provenance.json",
    "generalization_analysis.json",
    "phase4_report.json",
    "live_per_query.csv",
    "rankings.json",
    "candidate_pools.json",
    "failure_analysis.json",
    "routing_evaluation.json",
    "frozen_policy.json",
    "transfer_protocol.json",
)


class CandidateRecorder:
    """Observe the existing reranker trace without changing candidate generation or scoring."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.last_trace = None

    def search(self, query: str, top_k: int = 10):
        return self.search_with_trace(query, top_k).results

    def search_with_trace(self, query: str, top_k: int = 10):
        self.last_trace = self.retriever.search_with_trace(query, top_k)
        return self.last_trace


def external_systems(data, protocol: dict) -> tuple[dict, dict]:
    config = protocol["pipeline_config"]
    setup = {}
    start = perf_counter()
    encoder = SentenceEncoder(config["dense_path"], config["threads"], config["seed"])
    setup["dense_model_load_seconds"] = perf_counter() - start
    start = perf_counter()
    scorer = CrossEncoderScorer(config["rerank_path"], config["threads"], config["seed"])
    setup["reranker_model_load_seconds"] = perf_counter() - start
    identities = {"dense": encoder.identity, "reranker": scorer.identity}
    if protocol["reference_models"] and identities != protocol["reference_models"]:
        raise ValueError("Models/configuration no longer match the frozen Phase 3 source")
    start = perf_counter()
    bm25 = BM25Retriever(data.documents.values())
    setup["bm25_index_build_seconds"] = perf_counter() - start
    start = perf_counter()
    dense = DenseRetriever(
        data.documents.values(), encoder, config["batch_size"], Path(config["cache"])
    )
    setup["dense_index_setup_seconds"] = perf_counter() - start
    setup["dense_original_embedding_build_seconds"] = dense.original_build_seconds
    setup["dense_cache_status"] = dense.cache_status
    setup["dense_embedding_dimension"] = encoder.dimension
    hybrid = HybridRetriever(bm25, dense, config["candidate_depth"], config["rrf_constant"])
    reranked = RerankingRetriever(
        hybrid, data.documents.values(), scorer, config["rerank_candidates"], config["batch_size"]
    )
    setup["models"] = identities
    return dict(zip(METHODS, (bm25, dense, hybrid, reranked))), setup


def generalization(
    rows: list[dict],
    evaluation: dict,
    failure_summary: list[dict],
    phase2_report: dict,
    phase3_report: dict,
    k: int,
    seed: int,
) -> dict:
    current = {r["method"]: r for r in evaluation["summary"]}
    scifact2 = {r["method"]: r for r in phase2_report["summary"]}
    scifact3 = {r["method"]: r for r in phase3_report["routing_evaluation"]["summary"]}
    differences = {
        method: {
            f"{metric}@{k}": current[method][f"{metric}@{k}"] - scifact2[method][f"{metric}@{k}"]
            for metric in ("recall", "precision", "mrr", "ndcg")
        }
        for method in METHODS
    }
    by_method = {
        m: {r["query_id"]: r for r in rows if r["method"] == m} for m in (*METHODS, "router")
    }
    ids = sorted(by_method["bm25"])
    paired = {}
    for after, before in (("hybrid", "bm25"), ("reranked", "hybrid"), ("dense", "bm25")):
        paired[after + "_minus_" + before] = {
            key: paired_bootstrap(
                [by_method[after][q][key] for q in ids],
                [by_method[before][q][key] for q in ids],
                seed,
            )
            for key in (f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}", "latency_ms")
        }
    deltas = [
        by_method["reranked"][q][f"ndcg@{k}"] - by_method["hybrid"][q][f"ndcg@{k}"] for q in ids
    ]
    return {
        "nfcorpus_query_count": len(ids),
        "scifact_phase2_query_count": phase2_report["metadata"]["query_count"],
        "scifact_phase3_internal_query_count": len(
            phase3_report["routing_evaluation"]["query_ids"]
        ),
        "nfcorpus_summary": evaluation["summary"],
        "scifact_phase2_full_summary": phase2_report["summary"],
        "scifact_phase3_internal_summary": phase3_report["routing_evaluation"]["summary"],
        "absolute_nfcorpus_minus_scifact_full_metric_differences": differences,
        "absolute_nfcorpus_router_minus_scifact_internal_metrics": {
            f"{m}@{k}": current["router"][f"{m}@{k}"] - scifact3["router"][f"{m}@{k}"]
            for m in ("recall", "precision", "mrr", "ndcg")
        },
        "paired_within_nfcorpus": paired,
        "paired_router_within_nfcorpus": evaluation["paired_router_minus_baseline"],
        "reranker_query_effects": {
            "improved": sum(d > 0 for d in deltas),
            "degraded": sum(d < 0 for d in deltas),
            "unchanged": sum(d == 0 for d in deltas),
        },
        "nfcorpus_failure_summary": failure_summary,
        "scifact_full_failure_summary": phase3_report["failure_summary"],
        "route_fractions": evaluation["route_fractions"],
        "observed_differences": {
            "hybrid_minus_bm25_ndcg": current["hybrid"][f"ndcg@{k}"] - current["bm25"][f"ndcg@{k}"],
            "reranked_minus_hybrid_ndcg": current["reranked"][f"ndcg@{k}"]
            - current["hybrid"][f"ndcg@{k}"],
            "reranked_to_hybrid_mean_latency_ratio": current["reranked"]["mean_latency_ms"]
            / current["hybrid"]["mean_latency_ms"],
        },
        "comparison_scope": "Cross-dataset score subtraction is descriptive and UNPAIRED; no cross-dataset paired confidence interval. SciFact full has four methods, internal slice has five. Their sample sizes and dates differ.",
        "possible_explanations": [
            "Different query task and judgment density/grades can affect metrics and failure-indicator rates; not isolated causal effects.",
            "Corpus size, text length, truncation and CPU/runtime conditions may influence latency and quality.",
        ],
        "unsupported_claims": [
            "Universal generalization",
            "Causal explanation of individual failures",
            "Untouched pretraining test data",
            "Adaptive benefit from a hybrid-only policy",
        ],
        "limitations": "One additional biomedical collection, no retuning, single warmed CPU pass. Query bootstrap is descriptive and uncorrected for multiple comparisons, not repeated-run timing uncertainty.",
    }


def run_phase4(
    dataset_path: Path,
    protocol_path: Path,
    output: Path,
    phase2_report_path: Path,
    phase3_report_path: Path,
    systems_override: dict | None = None,
    plots: bool = True,
) -> Path:
    protocol_path, dataset_path = Path(protocol_path), Path(dataset_path)
    protocol, policy = load_protocol(protocol_path)  # Before reading external labels.
    config = protocol["pipeline_config"]
    k = config["k"]
    seed = config["seed"]
    run_id = (
        "phase4-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    root = Path(output) / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "frozen_policy.json").write_bytes((protocol_path / "frozen_policy.json").read_bytes())
    (root / "transfer_protocol.json").write_bytes((protocol_path / "protocol.json").read_bytes())
    metadata = {
        "run_id": run_id,
        "status": "initializing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "transfer_protocol_path": str(protocol_path.resolve()),
        "policy_record_sha256": protocol["policy_record_sha256"],
        "policy_file_sha256": protocol["policy_file_sha256"],
        "policy_frozen_before_external_data": protocol["created_at_utc"],
        "dataset_path": str(dataset_path.resolve()),
        "evaluation_split": protocol["evaluation_split"],
        "pipeline_config": config,
        "versions": versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "injected_systems": systems_override is not None,
        "events": [],
        "code_sha256": {
            p.relative_to(Path(__file__).resolve().parents[2]).as_posix(): file_hash(p)
            for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        },
        "timing": "Independent five-system execution, rotated order and one authored warmup each, CPU. Router features/decision included; model/index/statistics setup and metrics/I/O excluded. Source reranker traced through a lightweight observer.",
        "no_tuning": "No policy-selection or optimization API is used by Phase 4; exact saved Phase 3 policy imported before data loading.",
    }
    pools = {}

    def event(name):
        metadata["events"].append(
            {"name": name, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        )
        write_json(root / "experiment_metadata.json", metadata)

    event("verified_frozen_policy_before_dataset_load")
    try:
        provenance = read_json(dataset_path / "metadata.json")
        if provenance["transfer_policy_record_sha256"] != protocol["policy_record_sha256"]:
            raise ValueError("Dataset preparation used a different frozen policy")
        if datetime.fromisoformat(provenance["prepared_at_utc"]) < datetime.fromisoformat(
            protocol["created_at_utc"]
        ):
            raise ValueError("External data was prepared before this policy freeze")
        for name, expected in provenance["files_sha256"].items():
            if file_hash(dataset_path / name) != expected:
                raise ValueError("External data file changed")
        write_json(root / "dataset_provenance.json", provenance)
        data = load_dataset(dataset_path, protocol["evaluation_split"])
        ids = list(data.queries)
        metadata.update(
            {
                "query_ids": ids,
                "document_count": len(data.documents),
                "query_count": len(ids),
                "judgment_count": sum(map(len, data.qrels.values())),
                "dataset_files_sha256": {
                    **provenance["files_sha256"],
                    "metadata.json": file_hash(dataset_path / "metadata.json"),
                },
            }
        )
        event("external_evaluation_labels_loaded_after_policy_freeze")
        phase2_report, phase3_report = read_json(phase2_report_path), read_json(phase3_report_path)
        references = {
            "phase2_report_path": str(Path(phase2_report_path).resolve()),
            "phase2_report_sha256": file_hash(Path(phase2_report_path)),
            "phase3_report_path": str(Path(phase3_report_path).resolve()),
            "phase3_report_sha256": file_hash(Path(phase3_report_path)),
        }
        metadata["reference_reports"] = references
        start = perf_counter()
        corpus = CorpusFeatures(data.documents.values())
        metadata["corpus_feature_setup_seconds"] = perf_counter() - start
        if systems_override is None:
            systems, setup = external_systems(data, protocol)
            metadata["setup"] = setup
        else:
            systems = systems_override
        systems = {**systems, "reranked": CandidateRecorder(systems["reranked"])}
        metadata["status"] = "running"
        event("external_live_evaluation_started")

        def checkpoint(rows, decisions, rankings):
            q = rows[-1]["query_id"]
            trace = systems["reranked"].last_trace
            if trace is None:
                raise ValueError("Missing reranker candidate trace")
            pools[q] = {
                "hybrid": [asdict(r) for r in trace.candidates],
                "reranked": [asdict(r) for r in trace.reranked_candidates],
            }
            write_csv(root / "live_per_query.csv", rows)
            write_csv(root / "routing_decisions.csv", decisions)
            write_json(root / "rankings.json", rankings)
            write_json(root / "candidate_pools.json", pools)
            metadata["completed_queries"] = len(decisions)
            write_json(root / "experiment_metadata.json", metadata)

        rows, decisions, rankings = evaluate_live(data, ids, systems, corpus, policy, k, checkpoint)
        checkpoint(rows, decisions, rankings)
        load_protocol(protocol_path)
        if file_hash(root / "frozen_policy.json") != protocol["policy_file_sha256"]:
            raise ValueError("Policy mutated during evaluation")
        evaluation = evaluate_summary(rows, decisions, k, seed)
        evaluation["limitations"] = (
            "External NFCorpus test evaluation using an unchanged SciFact policy; no external tuning. Single warmed CPU measurement/query/system; paired query bootstrap does not capture runtime noise or prove broad generalization."
        )
        write_json(root / "routing_evaluation.json", evaluation)
        write_csv(root / "system_comparison.csv", evaluation["summary"])
        row_map = {(r["query_id"], r["method"]): r for r in rows}
        analysis = {}
        comparisons = []
        failure_rows = []
        for q in ids:
            typed = {
                m: [SearchResult(**entry) for entry in rankings[m][q]] for m in (*METHODS, "router")
            }
            typed_pool = [SearchResult(**entry) for entry in pools[q]["hybrid"]]
            records, facts = query_comparison(
                q,
                data.queries[q].text,
                typed,
                data.qrels.get(q, {}),
                typed_pool,
                {m: row_map[q, m] for m in typed},
                corpus,
                k,
            )
            comparisons.extend(records)
            analysis[q] = facts
            failure_rows.extend(
                {
                    "query_id": q,
                    "category": i["category"],
                    "evidence_type": i["evidence_type"],
                    "evidence": json.dumps(i["evidence"], sort_keys=True),
                }
                for i in facts["indicators"]
            )
        failure_summary = category_summary(analysis)
        write_csv(root / "per_query_comparison.csv", comparisons)
        write_csv(root / "failure_cases.csv", failure_rows)
        write_csv(root / "failure_summary.csv", failure_summary)
        write_json(root / "failure_analysis.json", analysis)
        gen = generalization(
            rows, evaluation, failure_summary, phase2_report, phase3_report, k, seed
        )
        write_json(root / "generalization_analysis.json", gen)
        report = {
            "run_id": run_id,
            "dataset": provenance,
            "policy_record_sha256": protocol["policy_record_sha256"],
            "policy_parameters": asdict(policy),
            "summary": evaluation["summary"],
            "route_fractions": evaluation["route_fractions"],
            "generalization": gen,
            "setup": metadata.get("setup", {}),
            "references": references,
            "unjudged_convention": "Zero evaluation gain, not known irrelevant",
            "tuning_performed": False,
        }
        write_json(root / "phase4_report.json", report)
        if plots:
            from app.evaluation.phase4_plots import generate_plots

            generate_plots(root)
        metadata.update(
            {
                "status": "complete",
                "payload_sha256": hashes(root, PAYLOAD),
                "required_artifacts": list(REQUIRED),
                "plots_enabled": plots,
            }
        )
        if plots:
            metadata["plot_sha256"] = hashes(
                root,
                [p.relative_to(root).as_posix() for p in sorted((root / "plots").glob("*.png"))],
            )
        event("complete_without_policy_update")
        replay_phase4(root, dataset_path)
        return root
    except BaseException as exc:
        metadata.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        event("failed")
        print(f"Incomplete Phase 4 run retained: {root}", file=sys.stderr)
        raise


def replay_phase4(root: Path, dataset_path: Path) -> int:
    root, dataset_path = Path(root), Path(dataset_path)
    metadata = read_json(root / "experiment_metadata.json")
    if metadata["status"] != "complete":
        raise ValueError("Cannot replay an incomplete external run")
    for group in ("payload_sha256", "plot_sha256"):
        if hashes(root, metadata.get(group, {})) != metadata.get(group, {}):
            raise ValueError("External artifact hash changed")
    protocol, policy = load_protocol(Path(metadata["transfer_protocol_path"]))
    if (root / "frozen_policy.json").read_bytes() != (
        Path(metadata["transfer_protocol_path"]) / "frozen_policy.json"
    ).read_bytes():
        raise ValueError("Run policy differs from frozen transfer source")
    if metadata["policy_record_sha256"] != protocol["policy_record_sha256"]:
        raise ValueError("Transfer policy identifier changed")
    if any(
        file_hash(dataset_path / name) != expected
        for name, expected in metadata["dataset_files_sha256"].items()
    ):
        raise ValueError("External dataset changed")
    data = load_dataset(dataset_path, metadata["evaluation_split"])
    ids = list(data.queries)
    k = metadata["pipeline_config"]["k"]
    if ids != metadata["query_ids"]:
        raise ValueError("External query set changed")
    names = (*METHODS, "router")
    rankings = read_json(root / "rankings.json")
    pools = read_json(root / "candidate_pools.json")
    if set(rankings) != set(names) or any(set(values) != set(ids) for values in rankings.values()):
        raise ValueError("Systems did not evaluate identical external queries")
    rows = read_csv(root / "live_per_query.csv")
    decisions = read_csv(root / "routing_decisions.csv")
    if len(rows) != len(ids) * 5 or {(r["query_id"], r["method"]) for r in rows} != {
        (q, m) for q in ids for m in names
    }:
        raise ValueError("Missing/duplicate external metric rows")
    corpus = CorpusFeatures(data.documents.values())
    if len(decisions) != len(ids) or {d["query_id"] for d in decisions} != set(ids):
        raise ValueError("Invalid routing query set")
    for decision in decisions:
        q = decision["query_id"]
        features = corpus.extract(data.queries[q].text)
        strategy, why = policy.decide(features)
        if (
            strategy != decision["strategy"]
            or why != decision["explanation"]
            or feature_dict(features) != json.loads(decision["features"])
        ):
            raise ValueError("Routing replay differs from label-free frozen decision")
        if rankings["router"][q] != rankings[strategy][q]:
            raise ValueError("Router ranks/scores differ from selected baseline")
    for row in rows:
        q, m = row["query_id"], row["method"]
        rs = rankings[m][q]
        if (
            len(rs) > k
            or [r["rank"] for r in rs] != list(range(1, len(rs) + 1))
            or any(
                r["document_id"] not in data.documents or not math.isfinite(r["score"]) for r in rs
            )
        ):
            raise ValueError("Invalid external ranking")
        expected = ranking_metrics([r["document_id"] for r in rs], data.qrels.get(q, {}), k)
        if any(
            not math.isclose(float(row[key]), value, abs_tol=1e-12)
            for key, value in expected.items()
        ):
            raise ValueError("Metric replay mismatch")
        for key in (*expected, "latency_ms", "candidate_count", "rerank_ms"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Invalid external numeric result")
        row["reranker_invoked"] = row["reranker_invoked"] == "True"
    if set(pools) != set(ids):
        raise ValueError("Candidate queries differ")
    for q in ids:
        before, after = pools[q]["hybrid"], pools[q]["reranked"]
        if sorted(r["document_id"] for r in before) != sorted(
            r["document_id"] for r in after
        ) or len(before) != len({r["document_id"] for r in before}):
            raise ValueError("Candidate IDs corrupted")
        if before[:k] != rankings["hybrid"][q] or after[:k] != rankings["reranked"][q]:
            raise ValueError("Candidate rank prefixes differ")
    evaluation = evaluate_summary(rows, decisions, k, metadata["pipeline_config"]["seed"])
    saved = read_json(root / "routing_evaluation.json")
    evaluation["limitations"] = saved["limitations"]
    if evaluation != saved:
        raise ValueError("Aggregate external evaluation does not replay")
    references = metadata["reference_reports"]
    for name in ("phase2", "phase3"):
        if (
            file_hash(Path(references[name + "_report_path"]))
            != references[name + "_report_sha256"]
        ):
            raise ValueError("SciFact reference changed")
    expected_gen = generalization(
        rows,
        evaluation,
        read_json(root / "phase4_report.json")["generalization"]["nfcorpus_failure_summary"],
        read_json(Path(references["phase2_report_path"])),
        read_json(Path(references["phase3_report_path"])),
        k,
        metadata["pipeline_config"]["seed"],
    )
    if expected_gen != read_json(root / "generalization_analysis.json"):
        raise ValueError("Generalization report does not replay")
    return len(rows)


def main() -> None:
    from app.tracking.mlflow import add_arguments, track_arguments

    parser = argparse.ArgumentParser(
        description="External NFCorpus evaluation with exact frozen Phase 3 router"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/public/nfcorpus"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/configs/phase4-transfer")
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/results/phase4"))
    parser.add_argument(
        "--scifact-phase2", type=Path, default=Path("experiments/reports/phase2_scifact_full.json")
    )
    parser.add_argument(
        "--scifact-phase3", type=Path, default=Path("experiments/reports/phase3_scifact.json")
    )
    parser.add_argument("--replay", type=Path)
    add_arguments(parser)
    args = parser.parse_args()
    try:
        if args.replay:
            print(f"Verified {replay_phase4(args.replay, args.dataset)} external system/query rows")
            track_arguments(args, args.replay, historical=True)
        else:
            source = run_phase4(
                args.dataset,
                args.protocol,
                args.output,
                args.scifact_phase2,
                args.scifact_phase3,
            )
            print(source)
            track_arguments(args, source)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Phase 4 failed: {exc}\n")


if __name__ == "__main__":
    main()
