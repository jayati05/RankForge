import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.data.loaders import load_dataset
from app.data.selection import select_queries
from app.evaluation.metrics import ranking_metrics
from app.retrieval.base import validate_top_k
from app.retrieval.bm25 import BM25Retriever


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    dataset_path: Path,
    output: Path,
    split: str = "test",
    k: int = 10,
    max_queries: int | None = None,
    seed: int = 42,
) -> Path:
    validate_top_k(k)
    dataset_path, output = Path(dataset_path), Path(output)
    full_data = load_dataset(dataset_path, split)
    data = select_queries(full_data, max_queries, seed)
    start = perf_counter()
    retriever = BM25Retriever(data.documents.values())
    build_seconds = perf_counter() - start
    rows, rankings = [], {}
    for qid, query in data.queries.items():
        start = perf_counter()
        results = retriever.search(query.text, k)
        latency_ms = (perf_counter() - start) * 1000
        rankings[qid] = [asdict(result) for result in results]
        rows.append(
            {
                "query_id": qid,
                "latency_ms": latency_ms,
                **ranking_metrics([r.document_id for r in results], data.qrels[qid], k),
            }
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    sources = [
        dataset_path / "corpus.jsonl",
        dataset_path / "queries.jsonl",
        dataset_path / "qrels" / f"{split}.tsv",
    ]
    if (dataset_path / "metadata.json").exists():
        sources.append(dataset_path / "metadata.json")
    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": data.metadata,
        "dataset_path": str(dataset_path.resolve()),
        "split": split,
        "document_count": len(data.documents),
        "query_count": len(rows),
        "available_query_count": len(full_data.queries),
        "subset": {
            "max_queries": max_queries,
            "seed": seed,
            "method": "sha256(seed:query_id), then lexicographic evaluation order",
        },
        "judgment_count": sum(map(len, data.qrels.values())),
        "query_ids": list(data.queries),
        "k": k,
        "system": "BM25Okapi",
        "config": {
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
            "tokenizer": "NFKC-casefold-identifier-regex-v1",
            "seed": None,
        },
        "index_build_seconds": build_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "versions": {name: version(name) for name in ["rank-bm25", "numpy"]},
        "source_sha256": {
            str(p.relative_to(dataset_path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "timing": "One first-pass search per query; excludes loading, index build and metrics. No warmup.",
        "limitations": "Single first-pass timing per query; no memory measurement. Unjudged gain=0 is an evaluation convention, not a claim of irrelevance.",
        "evaluation": {
            "aggregation": "macro mean across selected queries",
            "binary_relevance": "grade > 0",
            "ndcg_gain": "2**grade - 1",
            "unjudged_gain": 0,
            "precision_denominator": "k",
        },
        "code_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        },
    }
    summary = {
        "system": "bm25",
        "query_count": len(rows),
        **{key: mean(row[key] for row in rows) for key in rows[0] if key != "query_id"},
    }
    write_csv(run_dir / "per_query_results.csv", rows)
    write_csv(run_dir / "benchmark_summary.csv", [summary])
    for name, value in [("run_metadata.json", metadata), ("rankings.json", rankings)]:
        (run_dir / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return run_dir


def main() -> None:
    from app.tracking.mlflow import add_arguments, track_arguments

    parser = argparse.ArgumentParser(description="RankForge Phase 1 BM25 benchmark")
    parser.add_argument("--dataset", type=Path, default=Path("data/smoke"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--max-queries", type=int, help="Select this many queries; retain full corpus"
    )
    parser.add_argument("--seed", type=int, default=42, help="Stable SHA256 subset seed")
    add_arguments(parser)
    args = parser.parse_args()
    try:
        source = run_benchmark(
            args.dataset, args.output, args.split, args.k, args.max_queries, args.seed
        )
        print(source)
        track_arguments(args, source)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
