"""Transparent cross-run Phase6.2 aggregation, including rejected/failed attempts."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from app.evaluation.api_workload import aggregate, write_csv
from app.evaluation.soak_analysis import slope


def classification(checks):
    if not checks:
        return "Inconclusive"
    if any(v is False for v in checks.values()):
        return "Failed"
    if any(v is None for v in checks.values()):
        return "Inconclusive"
    return "Passed"


def latency(rows, outcome):
    selected = (
        [r for r in rows if r.get("status") == 429]
        if outcome == "rejected"
        else [r for r in rows if r["outcome"] == outcome]
    )
    return aggregate(selected, 1)["all_latency"]


def memory_summary(rows, stage, field):
    active = [r for r in rows if r["stage"] == stage and r.get(field) is not None]
    idle = [r for r in rows if r["stage"] == stage + "_idle" and r.get(field) is not None]
    if not active:
        return {"status": "unavailable"}
    end = active[-1]["elapsed_seconds"]
    late = [r for r in active if r["elapsed_seconds"] >= end - 120]
    return {
        "peak_mib": max(r[field] for r in active) / 2**20,
        "end_mib": active[-1][field] / 2**20,
        "whole_slope_mib_per_minute": slope([dict(r, rss_bytes=r[field]) for r in active]),
        "late_slope_mib_per_minute": slope([dict(r, rss_bytes=r[field]) for r in late]),
        "idle_start_mib": idle[0][field] / 2**20 if idle else None,
        "idle_end_mib": idle[-1][field] / 2**20 if idle else None,
        "idle_only_delta_mib": (idle[-1][field] - idle[0][field]) / 2**20 if idle else None,
        "late_trend_sufficient_duration": end - active[0]["elapsed_seconds"] >= 240,
    }


def variability(values):
    total = len(values)
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "missing": total, "status": "unavailable"}
    return {
        "n": len(values),
        "missing": total - len(values),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "sample_standard_deviation": stdev(values) if len(values) > 1 else None,
        "interpretation": "Descriptive independent-run variation; two repetitions do not establish a distribution or confidence interval.",
    }


def attribution(diagnostics, campaign_path):
    evidence = []
    for run in diagnostics:
        snapshots = run["snapshots"]
        first, last = snapshots[0], snapshots[-1]
        first_trace = (first.get("tracemalloc") or {}).get("current_bytes")
        last_trace = (last.get("tracemalloc") or {}).get("current_bytes")
        evidence.append(
            {
                "run": run["run"],
                "traced_python_current_delta_bytes": last_trace - first_trace
                if first_trace is not None and last_trace is not None
                else None,
                "tensor_counts": [
                    s.get("objects", {}).get("selected_types", {}).get("torch.Tensor")
                    for s in snapshots
                ],
                "logical_models_and_index": [
                    {
                        "label": s["label"],
                        "models": s.get("models"),
                        "dense_index_bytes": s.get("dense_index_bytes"),
                    }
                    for s in snapshots
                ],
                "application_references": [
                    {
                        "label": s["label"],
                        "tracked_requests": s.get("objects", {})
                        .get("selected_types", {})
                        .get("starlette.requests.Request", 0),
                        "tracked_json_responses": s.get("objects", {})
                        .get("selected_types", {})
                        .get("starlette.responses.JSONResponse", 0),
                    }
                    for s in snapshots
                ],
                "limits": "Tracemalloc started after initialization. GC census excludes untracked/native allocations; logical tensor sizes are not an accounting of all process memory.",
            }
        )
    controls = {}
    configuration = {}
    inputs = campaign_path / "analysis_inputs.json"
    if inputs.exists():
        configuration = json.loads(inputs.read_text())
        if configuration.get("blas_comparison"):
            controls["blas_import_kernel"] = json.loads(
                Path(configuration["blas_comparison"]).read_text()
            )
        if configuration.get("excluded_control_attempt"):
            controls["excluded_control_attempt"] = json.loads(
                Path(configuration["excluded_control_attempt"]).read_text()
            )
    inference = Path(
        configuration.get("inference_controls", str(campaign_path / "inference_controls"))
    )
    validity = inference / "validity_note.json"
    excluded = validity.exists() and json.loads(validity.read_text()).get(
        "excluded_from_primary_control_comparison"
    )
    if excluded:
        controls["excluded_control_attempt"] = json.loads(validity.read_text())
    if "blas_import_kernel" not in controls and all(
        (inference / f"blas_{n}/control.json").exists() for n in (1, 20)
    ):
        one = np.load(inference / "blas_1/scores.npy", allow_pickle=False)
        twenty = np.load(inference / "blas_20/scores.npy", allow_pickle=False)
        controls["blas_import_kernel"] = {
            "controls": {
                str(n): json.loads((inference / f"blas_{n}/control.json").read_text())
                for n in (1, 20)
            },
            "max_abs_score_delta": float(np.max(np.abs(one - twenty))),
            "identical_ranking": bool(np.array_equal(np.argsort(-one), np.argsort(-twenty))),
            "source": inference.as_posix(),
        }
    for label in () if excluded else ("hybrid", "reranked", "reranked_blas1"):
        path = inference / label / "operator_control.json"
        if path.exists():
            result = json.loads(path.read_text())
            controls[label] = {
                k: result[k]
                for k in (
                    "status",
                    "method",
                    "pid",
                    "command",
                    "query_id",
                    "versions",
                    "snapshots",
                    "configuration",
                    "design",
                )
            }
            controls[label]["profiler_status"] = result["profiler"]["status"]
            controls[label]["profiler_error"] = result["profiler"].get("error")
            controls[label]["top_self_operator_allocations"] = result["profiler"].get(
                "operators", []
            )[:5]
            controls[label]["source"] = path.as_posix()
    if (
        not excluded
        and (inference / "reranked/operator_control.json").exists()
        and (inference / "reranked_blas1/operator_control.json").exists()
    ):
        baseline = json.loads((inference / "reranked/operator_control.json").read_text())[
            "rankings"
        ]
        changed = json.loads((inference / "reranked_blas1/operator_control.json").read_text())[
            "rankings"
        ]
        same_ids_ranks = [(r["document_id"], r["rank"]) for r in baseline] == [
            (r["document_id"], r["rank"]) for r in changed
        ]
        delta = max(
            (abs(a["score"] - b["score"]) for a, b in zip(baseline, changed, strict=True)),
            default=0,
        )
        controls["reranked_blas_comparison"] = {
            "same_ordered_ids_ranks": same_ids_ranks,
            "max_absolute_score_delta": delta,
            "absolute_tolerance": 1e-5,
            "equivalent": same_ids_ranks and delta <= 1e-5,
        }
    return {
        "run_evidence": evidence,
        "controlled_probes": controls,
        "leak_status": "Inconclusive",
        "conclusion": "Native allocation stacks/heap ownership were not captured. Import/thread and fixed-query operator probes can identify contributors, but cannot rule out a leak or fully attribute long-run retention. No forced GC, allocator release or production memory optimization.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    campaign = json.loads((args.campaign / "campaign.json").read_text())
    root = Path.cwd()
    summaries = []
    times = []
    attempts = []
    diagnostics = []
    native_checkpoints = []
    comparisons = []
    failures = []
    for item in campaign["runs"]:
        if len(item["artifact_directories"]) != 1:
            failures.append(
                {"label": item["label"], "error": "Expected one artifact directory", "run": item}
            )
            continue
        path = root / item["artifact_directories"][0]
        report = json.loads((path / "phase6_1_report.json").read_text())
        if report["status"] != "complete":
            failures.append({"label": item["label"], "failure": report.get("failure")})
        memory = []
        for row in csv.DictReader((path / "memory_timeseries.csv").open()):
            memory.append(
                {k: v if k == "stage" else float(v) if v else None for k, v in row.items()}
            )
        req = []
        for row in csv.DictReader((path / "request_outcomes.csv").open()):
            row["latency_ms"] = float(row["latency_ms"])
            row["status"] = int(row["status"]) if row["status"] else None
            req.append(row)
            attempts.append({"run": item["label"], "run_id": report["run_id"], **row})
        times.extend(
            {
                "run": item["label"],
                "run_id": report["run_id"],
                "serving_pid": report["process_ids"]["serving"],
                **r,
            }
            for r in memory
        )
        snapshots = json.loads((path / "allocation_diagnostics.json").read_text())
        diagnostics.append({"run": item["label"], "snapshots": snapshots})
        replies = sorted(
            (
                p
                for p in path.glob("command_*.json")
                if json.loads(p.read_text()).get("command") == "snapshot"
            ),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        for snapshot, reply in zip(snapshots, replies, strict=True):
            native_checkpoints.append(
                {
                    "run": item["label"],
                    "run_id": report["run_id"],
                    "serving_pid": snapshot["pid"],
                    "stage": snapshot["label"],
                    "reply_published_utc": datetime.fromtimestamp(
                        reply.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    "native_threads": snapshot["native"]["counters"]["native_threads"],
                    "rss_bytes": snapshot["native"]["counters"]["rss_bytes"],
                    "private_commit_bytes": snapshot["native"]["counters"]["private_commit_bytes"],
                    **snapshot["native"]["regions"]["totals"],
                }
            )
        for stats, analysis in zip(report["summaries"], report["memory_analysis"], strict=True):
            rows = [r for r in req if r["stage"] == stats["stage"]]
            summary = {
                "run": item["label"],
                "run_id": report["run_id"],
                "serving_pid": report["process_ids"]["serving"],
                "config": report["config"],
                "workload_sha256": report["workload_sha256"],
                "statistics": stats,
                "rejection_latency": latency(rows, "rejected"),
                "budget_classification": classification(analysis["budgets"]["checks"]),
                "budgets": analysis["budgets"],
                "rss": memory_summary(memory, stats["stage"], "rss_bytes"),
                "private_commit": memory_summary(memory, stats["stage"], "private_commit_bytes"),
                "overall_sampled_peak_rss_mib": max(r["rss_bytes"] for r in memory) / 2**20,
                "overall_sampled_peak_private_commit_mib": max(
                    r["private_commit_bytes"] for r in memory
                )
                / 2**20,
                "os_lifetime_peak_rss_mib": max(
                    (
                        r["os_lifetime_peak_rss_bytes"] / 2**20
                        for r in memory
                        if r.get("os_lifetime_peak_rss_bytes") is not None
                    ),
                    default=None,
                ),
                "minimum_system_available_mib": min(
                    (
                        r["system_available_bytes"] / 2**20
                        for r in memory
                        if r.get("system_available_bytes") is not None
                    ),
                    default=None,
                ),
                "max_checkpoint_native_threads": max(
                    s["native"]["counters"]["native_threads"] for s in snapshots
                ),
                "sampling_scope": "RSS/private commit and native threads/virtual regions sampled continuously; full native maps at checkpoints."
                if any(r.get("native_threads") is not None for r in memory)
                else "RSS/private commit1s; native threads/virtual regions at stopped lifecycle checkpoints only. Missing continuous native sampler fields were detected; no invented or interpolated thread trace.",
                "graceful_shutdown": report.get("graceful_shutdown"),
                "correctness_checks": report["correctness_checks"],
                "sampler_errors": report["sampler_errors"],
            }
            summary["by_method"] = {}
            for method, method_stats in stats["by_method"].items():
                selected = [r for r in rows if r["method"] == method]
                cancelled = sum(r["outcome"] == "cancelled" for r in selected)
                rejected = method_stats["statuses"].get("429", 0)
                summary["by_method"][method] = {
                    **method_stats,
                    "cancelled": cancelled,
                    "other_errors": method_stats["failed"]
                    - rejected
                    - method_stats["timed_out"]
                    - cancelled,
                    "rejection_rate": rejected / len(selected) if selected else None,
                    "rejection_latency": latency(selected, "rejected"),
                }
            summaries.append(summary)
            comparisons.append(
                {
                    "run": item["label"],
                    "concurrency": stats["concurrency"],
                    "requests": stats["requests"],
                    "successful": stats["successful"],
                    "rejected": stats["statuses"].get("429", 0),
                    "rejection_rate": analysis["budgets"]["rejection_rate"],
                    "timed_out": stats["timed_out"],
                    "cancelled": stats["cancelled"],
                    "other_errors": analysis["budgets"]["other_errors"],
                    "successful_per_second": stats["successful_per_second"],
                    "attempts_per_second": stats["attempts_per_second"],
                    "success_p95_ms": stats["success_latency"]["p95_ms"],
                    "rejection_p95_ms": summary["rejection_latency"]["p95_ms"],
                    "rss_peak_mib": summary["rss"]["peak_mib"],
                    "commit_peak_mib": summary["private_commit"]["peak_mib"],
                    "rss_late_slope_mib_per_minute": summary["rss"]["late_slope_mib_per_minute"],
                    "commit_late_slope_mib_per_minute": summary["private_commit"][
                        "late_slope_mib_per_minute"
                    ],
                    "classification": summary["budget_classification"],
                }
            )

    def save(name, data):
        (args.campaign / name).write_text(
            json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
        )

    save("native_memory_summary.json", summaries)
    memory_attribution = attribution(diagnostics, args.campaign)
    save(
        "memory_attribution.json",
        {
            "observations": diagnostics,
            **memory_attribution,
        },
    )
    write_csv(args.campaign / "repeated_run_comparison.csv", comparisons)
    write_csv(args.campaign / "request_outcomes.csv", attempts)
    # Preserve actual observations and label their scope; never interpolate missing native fields.
    write_csv(args.campaign / "native_checkpoints.csv", native_checkpoints)
    thread_observations = [
        {
            **r,
            "elapsed_seconds": None,
            "native_sampling_ms": None,
            "observation_scope": "lifecycle_checkpoint",
        }
        for r in native_checkpoints
    ]
    thread_observations.extend(
        {**r, "reply_published_utc": None, "observation_scope": "continuous_native_sample"}
        for r in times
        if r.get("native_threads") is not None
    )
    write_csv(
        args.campaign / "thread_count_timeseries.csv",
        [
            {
                k: r.get(k)
                for k in (
                    "run",
                    "run_id",
                    "serving_pid",
                    "stage",
                    "reply_published_utc",
                    "native_threads",
                    "elapsed_seconds",
                    "native_sampling_ms",
                    "observation_scope",
                )
            }
            for r in thread_observations
        ],
    )
    for name, fields in [
        ("rss_timeseries.csv", ["rss_bytes"]),
        (
            "private_commit_timeseries.csv",
            [
                "private_commit_bytes",
            ],
        ),
    ]:
        write_csv(
            args.campaign / name,
            [
                {
                    k: r.get(k)
                    for k in ["run", "run_id", "serving_pid", "stage", "elapsed_seconds", *fields]
                }
                for r in times
            ],
        )
    pids = [s["serving_pid"] for s in summaries]
    variance = {
        str(c): {
            key: variability([r[key] for r in comparisons if r["concurrency"] == c])
            for key in (
                "success_p95_ms",
                "successful_per_second",
                "rss_peak_mib",
                "commit_peak_mib",
                "rss_late_slope_mib_per_minute",
                "commit_late_slope_mib_per_minute",
                "rejection_rate",
            )
        }
        for c in (1, 2)
    }
    final = {
        "campaign": campaign,
        "status": "complete" if not failures and campaign["status"] == "complete" else "failed",
        "runs": summaries,
        "budget_classification": classification(
            {
                s["run"]: True
                if s["budget_classification"] == "Passed"
                else False
                if s["budget_classification"] == "Failed"
                else None
                for s in summaries
            }
        ),
        "individual_budget_results": {
            s["run"]: {
                name: "Passed" if value is True else "Failed" if value is False else "Inconclusive"
                for name, value in s["budgets"]["checks"].items()
            }
            for s in summaries
        },
        "failures": failures,
        "independent_serving_pids": len(pids) == len(set(pids)),
        "same_workload_sha256": len({s["workload_sha256"] for s in summaries}) == 1,
        "variability": variance,
        "memory_attribution": memory_attribution,
        "full_duration_design_satisfied": len(summaries) == 5
        and sum(s["statistics"]["concurrency"] == 1 for s in summaries) == 2
        and sum(s["statistics"]["concurrency"] == 2 for s in summaries) == 2
        and sum(
            s["statistics"]["concurrency"] == 4 and s["statistics"]["stress"] for s in summaries
        )
        == 1
        and all(
            s["config"]["warmup"] == 60
            and s["config"]["idle"] == 300
            and s["config"].get("sample_interval") == 1
            and s["config"].get("timeout") == 30
            and s["config"].get("seed") == 42
            and s["statistics"]["submission_duration_seconds"]
            == (300 if s["statistics"]["stress"] else 900)
            for s in summaries
        ),
        "limitations": "Two repetitions per normal level; one stress run. Closed-loop duration-based workload does not guarantee equal per-method completion or the same query prefix length. Profiling perturbs execution. Traced Python memory cannot be subtracted from RSS/private commit to obtain a native leak size. No native allocation stacks, no proof of lifetime boundedness. All attempts including429/timeouts/errors stay in denominators.",
    }
    save("phase6_2_report.json", final)
    figures(args.campaign, times, comparisons, attempts)
    print(
        json.dumps(
            {"status": final["status"], "campaign": str(args.campaign), "runs": len(summaries)}
        )
    )


def figures(path, times, comparison, attempts):
    os.environ["MPLCONFIGDIR"] = str(path / ".matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(r["run"] for r in times))
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 3 * len(labels)), squeeze=False)
    for label, ax in zip(labels, axes[:, 0], strict=True):
        rows = [r for r in times if r["run"] == label]
        for field in ("rss_bytes", "private_commit_bytes"):
            ax.plot(
                [r["elapsed_seconds"] / 60 for r in rows],
                [r[field] / 2**20 for r in rows],
                label={"rss_bytes": "RSS / working set", "private_commit_bytes": "Private commit"}[
                    field
                ],
            )
        idle_rows = [r for r in rows if r["stage"].endswith("_idle")]
        if idle_rows:
            ax.axvspan(
                idle_rows[0]["elapsed_seconds"] / 60,
                idle_rows[-1]["elapsed_seconds"] / 60,
                color="grey",
                alpha=0.12,
                label="Idle",
            )
        ax.set(title=label, xlabel="Process sampling minutes", ylabel="MiB")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path / "rss_vs_commit.png")
    plt.close(fig)
    for idle, name in [(False, "rss_across_runs.png"), (True, "idle_recovery.png")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label in labels:
            rows = [
                r
                for r in times
                if r["run"] == label and (r["stage"].endswith("_idle") if idle else True)
            ]
            if rows:
                ax.plot(
                    [(r["elapsed_seconds"] - rows[0]["elapsed_seconds"]) / 60 for r in rows],
                    [r["rss_bytes"] / 2**20 for r in rows],
                    label=label,
                )
        ax.set(xlabel="Minutes since first included sample", ylabel="RSS MiB")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path / name)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in comparison:
        ax.scatter(r["concurrency"], r["success_p95_ms"], label=r["run"])
    ax.axhline(5000, color="red", linestyle="--", label="Unchanged 5,000 ms budget")
    ax.set(xlabel="Concurrency", ylabel="Successful request p95 (ms)", xticks=[1, 2, 4])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "latency_vs_concurrency.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = sorted({r["method"] for r in attempts})
    for i, label in enumerate(labels):
        rates = []
        for method in methods:
            rows = [
                r
                for r in attempts
                if r["run"] == label and r["method"] == method and r["stage"] != "warmup"
            ]
            rates.append(sum(r["status"] == 429 for r in rows) / len(rows) if rows else 0)
        ax.bar(np.arange(len(methods)) + i * 0.15, rates, width=0.15, label=label)
    ax.set(xticks=np.arange(len(methods)) + 0.3, xticklabels=methods, ylabel="429/all attempts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "rejection_by_method.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
