"""Small isolated hybrid/reranked inference memory control, separate from HTTP soaks."""

import argparse
from dataclasses import asdict
import json
from importlib.metadata import version
import os
from pathlib import Path
import sys
import time

from app.evaluation.windows_native import counters, regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("hybrid", "reranked"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from app.core.config import Settings
    from app.services.model_service import ModelService
    from app.evaluation.api_soak import query_pool
    from app.evaluation.native_diagnostics import configuration
    import torch

    snapshots = []
    started = time.perf_counter()

    def snapshot(label):
        snapshots.append(
            {
                "label": label,
                "elapsed_seconds": time.perf_counter() - started,
                **counters(os.getpid()),
                "region_totals": regions(os.getpid())["totals"],
            }
        )
        (args.output / "progress.json").write_text(
            json.dumps({"stage": label, "snapshots": snapshots}), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "stage": label,
                    "elapsed_seconds": snapshots[-1]["elapsed_seconds"],
                    "pid": os.getpid(),
                }
            ),
            flush=True,
        )

    snapshot("before_service")
    service = ModelService(Settings())
    try:
        for name in ("bm25", "dense", "hybrid", "reranked", "router"):
            service.get(name)
            print(
                json.dumps({"initialized": name, "elapsed_seconds": time.perf_counter() - started}),
                flush=True,
            )
        retriever = service.get(args.method)
        query_record = query_pool(None)[0]
        query = query_record["text"]
        snapshot("all_resources_initialized")
        expected = [asdict(hit) for hit in retriever.search(query, 10)]
        for _ in range(5):
            retriever.search(query, 10)
        snapshot("after_5_warmup")
        rows = []
        for i in range(40):
            before = time.perf_counter()
            hits = [asdict(h) for h in retriever.search(query, 10)]
            if any(
                a["document_id"] != b["document_id"]
                or a["rank"] != b["rank"]
                or abs(a["score"] - b["score"]) > 1e-5
                for a, b in zip(hits, expected, strict=True)
            ):
                raise AssertionError("Repeated inference ranking mismatch")
            rows.append(
                {
                    "iteration": i + 1,
                    "latency_ms": (time.perf_counter() - before) * 1000,
                    **counters(os.getpid()),
                }
            )
            if (i + 1) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "completed_iterations": i + 1,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        snapshot("after_40_repeated_calls")
        profiler = {}
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=True,
            ) as profile:
                retriever.search(query, 10)
            operators = [
                {
                    "operator": event.key,
                    "count": event.count,
                    "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                    "inclusive_cpu_memory_bytes": event.cpu_memory_usage,
                    "self_cpu_time_us": event.self_cpu_time_total,
                }
                for event in profile.key_averages()
            ]
            if not operators:
                raise RuntimeError("CPU profiler returned no operator observations")
            profiler = {
                "status": "complete",
                "operators": sorted(operators, key=lambda r: -r["self_cpu_memory_bytes"]),
                "limitations": "One post-loop CPU call; profiler introduces allocations and may miss pre-existing blocks. Operator allocation traffic/net counts are not RSS or a retained-heap attribution. Inclusive counts overlap.",
            }
            (args.output / "operator_memory.txt").write_text(
                profile.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=25),
                encoding="utf-8",
            )
        except Exception as exc:
            profiler = {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        snapshot("after_separate_profiler_call")
        report = {
            "status": "complete",
            "method": args.method,
            "pid": os.getpid(),
            "command": [sys.executable, "-m", "app.evaluation.native_operator", *sys.argv[1:]],
            "query": query,
            "query_id": query_record["id"],
            "versions": {
                name: version(name)
                for name in (
                    "torch",
                    "numpy",
                    "sentence-transformers",
                    "transformers",
                    "threadpoolctl",
                )
            },
            "iterations": 40,
            "warmup_iterations": 5,
            "snapshots": snapshots,
            "requests": rows,
            "rankings": expected,
            "configuration": configuration(),
            "resources": service.snapshot(),
            "profiler": profiler,
            "design": "Fresh process per condition; same all-resource initialization/seed/query/k10/candidate settings/40repetitions. Hybrid vs reranked changes only selected pipeline; reranked vs reranked_blas1 changes only child OPENBLAS_NUM_THREADS. Actual configuration is recorded. No forced GC, allocator release or HTTP traffic; no quality evaluation or load-latency claim.",
        }
        (args.output / "operator_control.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "profiler_status": profiler["status"],
                    "pid": os.getpid(),
                }
            )
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
