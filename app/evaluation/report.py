"""Export measured comparison summaries and observed disagreement examples."""

import argparse
import json
from pathlib import Path

from app.evaluation.comparison import replay_comparison, write_json


def export_report(run_dir: Path, dataset_path: Path, output: Path) -> dict:
    replay_comparison(run_dir, dataset_path)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    failures = json.loads((run_dir / "failure_cases.json").read_text(encoding="utf-8"))
    k = metadata["config"]["k"]
    examples = {}
    comparisons = [("rerank_vs_hybrid", "reranked", "hybrid"), ("dense_vs_bm25", "dense", "bm25")]
    for label, after, before in comparisons:
        differences = [
            (qid, value["metrics"][after][f"ndcg@{k}"] - value["metrics"][before][f"ndcg@{k}"])
            for qid, value in failures.items()
        ]
        for direction in ("improvement", "regression"):
            eligible = [
                (qid, delta)
                for qid, delta in differences
                if (delta > 0 if direction == "improvement" else delta < 0)
            ]
            eligible.sort(
                key=lambda item: (-item[1] if direction == "improvement" else item[1], item[0])
            )
            if eligible:
                qid, delta = eligible[0]
                examples[label + "_" + direction] = {
                    "query_id": qid,
                    "ndcg_delta": delta,
                    "observation": failures[qid],
                }
            else:
                examples[label + "_" + direction] = None
    report = {
        "artifact_directory": str(run_dir),
        "metadata": metadata,
        "summary": summary,
        "disagreement_query_count": len(failures),
        "rerank_effect_counts_among_disagreements": {
            effect: sum(value["rerank_effect"] == effect for value in failures.values())
            for effect in ("improved", "degraded", "unchanged")
        },
        "examples": examples,
        "limitations": "Examples selected by largest absolute nDCG difference in this run, ties by query ID. Descriptive observations, not causal explanations. Missing example means no such case was observed.",
    }
    if Path(output).exists():
        raise FileExistsError(f"Refusing report overwrite: {output}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(output), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a replay-verified Phase 2 report")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/public/scifact"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = export_report(args.run, args.dataset, args.output)
        print(
            json.dumps({"report": str(args.output), "queries": report["metadata"]["query_count"]})
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Report failed: {exc}\n")


if __name__ == "__main__":
    main()
