"""Phase 6.1 fixed-duration soaks with idle recovery in the same API process."""

import argparse
import asyncio
import csv
import json
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx

from app.evaluation.api_load import (
    ROOT,
    compare,
    direct,
    post_control,
    save,
    serving_pid,
    sha,
    wait_idle,
)
from app.evaluation.api_workload import (
    METHODS,
    MemorySampler,
    aggregate,
    fingerprint,
    queries,
    schedule,
    weights,
    write_csv,
)
from app.evaluation.soak_analysis import analyze, budgets
from app.evaluation.soak_diagnostics import system_memory


def query_pool(path: Path | None) -> list[dict]:
    pool = (
        json.loads(path.read_text(encoding="utf-8"))
        if path
        else queries(ROOT / "data/public/scifact")
    )
    if (
        not pool
        or len({q["id"] for q in pool}) != len(pool)
        or any(
            not isinstance(q["id"], str)
            or not isinstance(q["text"], str)
            or not q["text"].strip()
            or len(q["text"]) > 1024
            for q in pool
        )
    ):
        raise ValueError("Query set requires unique string IDs and nonblank text <=1024 characters")
    return pool


def reference(output, pool):
    from app.data.loaders import read_jsonl
    from app.data.schemas import Document
    from app.evaluation.features import CorpusFeatures
    from app.evaluation.transfer_policy import load_protocol

    direct(output, pool)
    data = json.loads(output.read_text())
    features = CorpusFeatures(
        read_jsonl(ROOT / "data/public/scifact/corpus.jsonl", Document).values()
    )
    _, policy = load_protocol(ROOT / "experiments/configs/phase4-transfer")
    data["routing_decisions"] = {
        q["id"]: policy.decide(features.extract(q["text"]))[0] for q in pool
    }
    save(output, data)


class SoakSampler(MemorySampler):
    def __init__(self, pid, interval):
        super().__init__(pid, interval)
        self._sample_lock = threading.Lock()

    def sample(self):
        with self._sample_lock:
            self._sample()

    def _sample(self):
        before = len(self.rows)
        super().sample()
        if len(self.rows) > before:
            try:
                self.rows[-1].update(system_memory())
            except OSError:
                self.rows[-1].update(system_total_bytes=None, system_available_bytes=None)


async def duration_batch(
    url, cycle, duration, concurrency, timeout, stage, expected, progress, sink
):
    """Stop submissions at deadline; drain already submitted calls, never force-cancel them."""
    rows = []
    next_index = 0
    async with httpx.AsyncClient(
        base_url=url,
        timeout=timeout,
        trust_env=False,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
    ) as client:
        start = time.perf_counter()

        async def worker():
            nonlocal next_index
            while time.perf_counter() - start < duration:
                index = next_index
                next_index += 1
                item = cycle[index % len(cycle)]
                began = time.perf_counter()
                row = {
                    "stage": stage,
                    "request_index": index,
                    "method": item["method"],
                    "query_id": item["query"]["id"],
                    "start_seconds": began - start,
                    "end_seconds": None,
                    "latency_ms": None,
                    "status": None,
                    "outcome": "client_error",
                    "error_code": None,
                    "request_id": None,
                    "correct": None,
                    "selected_strategy": None,
                }
                try:
                    response = await client.post(
                        "/search",
                        json={
                            "query": item["query"]["text"],
                            "method": item["method"],
                            "top_k": 10,
                        },
                    )
                    row["latency_ms"] = (time.perf_counter() - began) * 1000
                    row["status"] = response.status_code
                    body = response.json()
                    row["request_id"] = body.get("request_id")
                    if response.status_code == 200:
                        row["outcome"] = "success"
                        row["selected_strategy"] = body["selected_strategy"]
                        row["correct"] = compare(
                            body["documents"],
                            expected["rankings"][item["method"]][item["query"]["id"]],
                        )
                        if item["method"] == "router":
                            row["correct"] = (
                                row["correct"]
                                and body["routing"]["policy_record_sha256"]
                                == expected["policy_record_sha256"]
                                and body["selected_strategy"]
                                == expected["routing_decisions"][item["query"]["id"]]
                            )
                    else:
                        row.update(
                            outcome="http_error", error_code=body.get("error", {}).get("code")
                        )
                except httpx.TimeoutException as exc:
                    row.update(outcome="timeout", error_code=type(exc).__name__)
                except asyncio.CancelledError:
                    row.update(outcome="cancelled", error_code="CancelledError")
                    raise
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    row.update(outcome="client_error", error_code=type(exc).__name__)
                finally:
                    row["end_seconds"] = time.perf_counter() - start
                    if row["latency_ms"] is None:
                        row["latency_ms"] = (time.perf_counter() - began) * 1000
                    rows.append(row)
                    sink(row)

        async def heartbeat(tasks):
            while not all(t.done() for t in tasks):
                elapsed = time.perf_counter() - start
                save(
                    progress,
                    {
                        "stage": stage,
                        "elapsed_seconds": elapsed,
                        "duration_seconds": duration,
                        "requests": len(rows),
                        "successful": sum(r["outcome"] == "success" for r in rows),
                        "rejected": sum(r["status"] == 429 for r in rows),
                    },
                )
                await asyncio.sleep(5)

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        monitor = asyncio.create_task(heartbeat(tasks))
        try:
            await asyncio.gather(*tasks)
        finally:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
    elapsed = time.perf_counter() - start
    stats = aggregate(rows, elapsed)
    stats["by_method"] = {
        method: aggregate([r for r in rows if r["method"] == method], elapsed) for method in METHODS
    }
    stats["cancelled"] = sum(r["outcome"] == "cancelled" for r in rows)
    stats["submission_duration_seconds"] = duration
    stats["drain_overrun_seconds"] = max(0, elapsed - duration)
    return rows, stats


def figures(output, resources, requests, stages):
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(
        [r["elapsed_seconds"] / 60 for r in resources], [r["rss_bytes"] / 2**20 for r in resources]
    )
    ax.set(
        xlabel="Serving-process observation (minutes)",
        ylabel="Working set / RSS (MiB)",
        title="Phase 6.1: same serving process across soaks and idle periods",
    )
    fig.tight_layout()
    fig.savefig(output / "memory_vs_time.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    for stage in stages:
        rows = [r for r in requests if r["stage"] == stage]
        ax.scatter(
            [r["start_seconds"] / 60 for r in rows],
            [r["latency_ms"] for r in rows],
            s=2,
            label=stage,
            alpha=0.5,
        )
    ax.set(
        xlabel="Minutes within each stage",
        ylabel="All-attempt HTTP latency (ms)",
        title="Includes 429/errors; successful percentiles in summary",
    )
    if ax.has_data():
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "latency_vs_time.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    success = [
        sum(
            r["method"] == m and r["outcome"] == "success" for r in requests if r["stage"] in stages
        )
        for m in METHODS
    ]
    failure = [
        sum(
            r["method"] == m and r["outcome"] != "success" for r in requests if r["stage"] in stages
        )
        for m in METHODS
    ]
    ax.bar(METHODS, success, label="Success")
    ax.bar(METHODS, failure, bottom=success, label="Non-success")
    ax.set(ylabel="Measured requests", title="Request outcomes; all non-successes retained")
    if ax.has_data():
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "outcomes_by_method.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    for stage in stages:
        rows = [r for r in resources if r["stage"] == stage + "_idle"]
        if rows:
            ax.plot(
                [(r["elapsed_seconds"] - rows[0]["elapsed_seconds"]) / 60 for r in rows],
                [r["rss_bytes"] / 2**20 for r in rows],
                label=stage,
            )
    ax.set(
        xlabel="Idle minutes after drained workload",
        ylabel="Working set / RSS (MiB)",
        title="Recovery without forced garbage collection or cache clearing",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "idle_recovery.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=900, help="Seconds per normal soak")
    parser.add_argument(
        "--clients", default="1,2", help="Sequential fixed levels in the same server"
    )
    parser.add_argument(
        "--stress-duration",
        type=float,
        default=300,
        help="0 disables optional concurrency-4 stress",
    )
    parser.add_argument("--idle", type=float, default=300)
    parser.add_argument("--warmup", type=float, default=60)
    parser.add_argument("--sample-interval", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", default=",".join(m + ":1" for m in METHODS))
    parser.add_argument("--query-set", type=Path)
    parser.add_argument("--window", type=float, default=120)
    parser.add_argument("--rss-budget-mib", type=float, default=2048)
    parser.add_argument(
        "--growth-budget", type=float, default=2, help="MiB/min allowed late-window RSS slope"
    )
    parser.add_argument("--rejection-budget", type=float, default=0.05)
    parser.add_argument("--stress-rejection-budget", type=float, default=1)
    parser.add_argument("--p95-budget-ms", type=float, default=5000)
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--native-profile", action="store_true", help="Opt-in Windows Phase6.2 diagnostics"
    )
    parser.add_argument("--serve", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--control-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.native_profile:
        from app.evaluation.native_diagnostics import install

        install(sys.modules[__name__])
    if args.serve:
        from app.evaluation.soak_diagnostics import serve

        serve(args.serve, args.control_file)
        return
    pool = query_pool(args.query_set)
    if args.reference:
        reference(args.reference, pool)
        return
    levels = [int(n) for n in args.clients.split(",") if n]
    distribution = weights(args.methods)
    if not all(
        math.isfinite(value)
        for value in (
            args.duration,
            args.idle,
            args.warmup,
            args.stress_duration,
            args.sample_interval,
            args.timeout,
            args.window,
            args.rss_budget_mib,
            args.growth_budget,
            args.rejection_budget,
            args.stress_rejection_budget,
            args.p95_budget_ms,
        )
    ):
        parser.error("Durations, intervals and budgets must be finite")
    if set(distribution) != set(METHODS):
        parser.error("All five methods must have explicit positive weights")
    if (
        args.duration <= 0
        or args.idle < 0
        or args.warmup < 0
        or args.stress_duration < 0
        or not 0.1 <= args.sample_interval <= 60
        or not 0 < args.timeout <= 600
        or (not levels and not args.stress_duration)
        or any(n < 1 or n > 16 for n in levels)
        or args.window <= 0
        or args.rss_budget_mib <= 0
        or args.growth_budget < 0
        or not 0 <= args.rejection_budget <= 1
        or not 0 <= args.stress_rejection_budget <= 1
        or args.p95_budget_ms <= 0
    ):
        parser.error("Invalid duration/concurrency/interval/timeout/budget configuration")
    cycle = schedule(pool, 40 * sum(distribution.values()), distribution, args.seed)
    phase = "phase6_2" if args.native_profile else "phase6_1"
    run_id = (
        phase + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    )
    output = ROOT / "experiments/results" / phase / run_id
    output.mkdir(parents=True)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("serve", "control_file", "reference")
    }
    report = {
        "run_id": run_id,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "artifact_directory": output.relative_to(ROOT).as_posix(),
        "command": subprocess.list2cmdline(
            [sys.executable, "-m", "app.evaluation.api_soak", *sys.argv[1:]]
        ),
        "workload_sha256": fingerprint(cycle),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "versions": {
            p: version(p)
            for p in (
                "fastapi",
                "starlette",
                "uvicorn",
                "httpx",
                "numpy",
                "torch",
                "transformers",
                "sentence-transformers",
                "matplotlib",
            )
        },
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p)
            for base in ("app", "scripts")
            for p in (ROOT / base).rglob("*.py")
        },
        "budgets_note": "Pre-registered local research targets; not universal production thresholds. Stress permits rejection intentionally; normal <= configured fraction. Run completion is separate from budget pass.",
    }
    save(output / "phase6_1_report.json", report)
    save(output / "workload.json", {"queries": pool, "cycle": cycle})
    query_file = output / "query_set.json"
    save(query_file, pool)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RANKFORGE_")}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    stages = [(f"soak_{i + 1}_c{c}", c, args.duration, False) for i, c in enumerate(levels)]
    if args.stress_duration:
        stages.append(("stress_c4", 4, args.stress_duration, True))
    control_file = output / "control.json"
    post_control(control_file, 0, "idle")
    sequence = 0
    child = sampler = None
    requests = []
    checks = []
    summary = []
    analysis = []
    snapshots = []
    lifecycle = []
    stream = (output / "request_outcomes.csv").open("w", newline="", encoding="utf-8")
    writer = None

    def sink(row):
        nonlocal writer
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        stream.flush()

    def control(command):
        nonlocal sequence
        sequence += 1
        post_control(control_file, sequence, command)
        reply = output / f"command_{sequence}.json"
        deadline = time.monotonic() + 900
        while True:
            if reply.exists():
                try:
                    data = json.loads(reply.read_text())
                    break
                except (OSError, ValueError):
                    pass
            if child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Benchmark control did not complete")
            threading.Event().wait(0.05)
        assert data["status"] == "complete", data
        return data

    try:
        with (output / "direct.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "app.evaluation.api_soak",
                    "--reference",
                    str(output / "direct_rankings.json"),
                    "--query-set",
                    str(query_file),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                check=True,
            )
        expected = json.loads((output / "direct_rankings.json").read_text())
        report["policy_record_sha256"] = expected["policy_record_sha256"]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with (output / "server.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "app.evaluation.api_soak",
                    "--serve",
                    str(port),
                    "--control-file",
                    str(control_file),
                    *(["--native-profile"] if args.native_profile else []),
                ],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            url = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=url, timeout=900, trust_env=False) as client:
                deadline = time.monotonic() + 60
                while True:
                    try:
                        cold = client.get("/health").json()
                        break
                    except httpx.TransportError:
                        if child.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError("Server startup failed")
                        threading.Event().wait(0.1)
                pid = serving_pid((output / "server.log").read_text())
                report["process_ids"] = {"launcher": child.pid, "serving": pid}
                assert all(r["initialization_count"] == 0 for r in cold["resources"].values())
                sampler = SoakSampler(pid, args.sample_interval)
                sampler.thread.start()
                sampler.mark("before_initialization")
                control("initialize")
                sampler.mark("after_initialization")
                if args.trace:
                    control("trace_start")
                if args.warmup:
                    rows, _ = asyncio.run(
                        duration_batch(
                            url,
                            cycle,
                            args.warmup,
                            1,
                            args.timeout,
                            "warmup",
                            expected,
                            output / "progress.json",
                            sink,
                        )
                    )
                    requests.extend(rows)
                wait_idle(client, 900)
                sampler.mark("after_warmup")

                def snapshot(label):
                    data = control("snapshot")["diagnostics"]
                    data["label"] = label
                    snapshots.append(data)
                    save(output / "allocation_diagnostics.json", snapshots)

                def correctness(label):
                    for method in METHODS:
                        for query in pool:
                            response = client.post(
                                "/search",
                                json={"query": query["text"], "method": method, "top_k": 10},
                            )
                            correct = response.status_code == 200 and compare(
                                response.json()["documents"],
                                expected["rankings"][method][query["id"]],
                            )
                            if correct and method == "router":
                                correct = (
                                    response.json()["routing"]["policy_record_sha256"]
                                    == expected["policy_record_sha256"]
                                    and response.json()["selected_strategy"]
                                    == expected["routing_decisions"][query["id"]]
                                )
                            checks.append(
                                {
                                    "label": label,
                                    "method": method,
                                    "query_id": query["id"],
                                    "passed": correct,
                                    "status": response.status_code,
                                }
                            )
                            if not correct:
                                raise AssertionError("Correctness mismatch")

                correctness("initial")
                snapshot("initial_after_warmup")
                for stage, concurrency, duration, stress in stages:
                    sampler.mark(stage)
                    print(
                        json.dumps({"stage": stage, "duration_seconds": duration, "pid": pid}),
                        flush=True,
                    )
                    rows, stats = asyncio.run(
                        duration_batch(
                            url,
                            cycle,
                            duration,
                            concurrency,
                            args.timeout,
                            stage,
                            expected,
                            output / "progress.json",
                            sink,
                        )
                    )
                    requests.extend(rows)
                    if any(r["correct"] is False for r in rows):
                        raise AssertionError("Sustained correctness mismatch")
                    stats.update(stage=stage, concurrency=concurrency, stress=stress)
                    summary.append(stats)
                    wait_idle(client, 900)
                    sampler.mark(stage + "_stopped")
                    snapshot(stage + "_stopped")
                    sampler.mark(stage + "_idle")
                    idle_start = time.perf_counter()
                    while time.perf_counter() - idle_start < args.idle:
                        save(
                            output / "progress.json",
                            {
                                "stage": stage + "_idle",
                                "elapsed_seconds": time.perf_counter() - idle_start,
                                "duration_seconds": args.idle,
                                "requests": stats["requests"],
                                "successful": stats["successful"],
                                "rejected": stats["statuses"].get("429", 0),
                            },
                        )
                        threading.Event().wait(
                            min(5, max(0, args.idle - (time.perf_counter() - idle_start)))
                        )
                    sampler.sample()
                    sampler.mark(stage + "_recovery_checks")
                    snapshot(stage + "_after_idle")
                    correctness(stage + "_after_idle")
                    health = client.get("/health").json()
                    assert health["status"] == "ok"
                    assert all(
                        r["initialization_count"] == 1 and r["initialization_attempts"] == 1
                        for r in health["resources"].values()
                    )
                    lifecycle.append(
                        {"stage": stage, "health": health, "same_pid": pid, "healthy": True}
                    )
                    result = analyze(sampler.rows, stage, args.window, args.growth_budget)
                    result["budgets"] = budgets(
                        stats,
                        result,
                        args.rss_budget_mib,
                        args.stress_rejection_budget if stress else args.rejection_budget,
                        args.p95_budget_ms,
                    )
                    analysis.append(result)
                    save(output / "soak_summary.json", summary)
                    save(output / "memory_analysis.json", analysis)
                    save(output / "correctness_checks.json", checks)
                    save(output / "lifecycle_summary.json", lifecycle)
                    print(
                        json.dumps(
                            {
                                "completed": stage,
                                "requests": stats["requests"],
                                "successful": stats["successful"],
                                "budgets": result["budgets"],
                            }
                        ),
                        flush=True,
                    )
                control("stop")
                child.wait(timeout=120)
                assert child.returncode == 0
        report["status"] = "complete"
        report["graceful_shutdown"] = "exit0; lifespan release/service-stop events checked"
        events = [
            json.loads(line)
            for line in (output / "server.log").read_text().splitlines()
            if line.startswith("{")
        ]
        assert any(e.get("event") == "service_stopped" for e in events)
    except BaseException as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if child and child.poll() is None:
            try:
                control("stop")
                child.wait(timeout=120)
            except (Exception, KeyboardInterrupt):
                child.terminate()
                child.wait(timeout=30)
                report["forced_cleanup"] = True
        if sampler:
            sampler.finish()
        stream.close()
        resources = sampler.rows if sampler else []
        if resources:
            write_csv(output / "memory_timeseries.csv", resources)
        if requests:
            write_csv(
                output / "latency_timeseries.csv",
                [
                    {
                        k: r[k]
                        for k in (
                            "stage",
                            "request_index",
                            "method",
                            "start_seconds",
                            "end_seconds",
                            "latency_ms",
                            "status",
                            "outcome",
                        )
                    }
                    for r in requests
                ],
            )
        save(output / "soak_summary.json", summary)
        save(output / "memory_analysis.json", analysis)
        save(
            output / "correctness_checks.json",
            {
                "pre_post": checks,
                "during_checked": sum(r["correct"] is not None for r in requests),
                "during_failures": sum(r["correct"] is False for r in requests),
                "absolute_tolerance": 1e-5,
            },
        )
        save(
            output / "lifecycle_summary.json",
            {
                "checkpoints": lifecycle,
                "process_ids": report.get("process_ids"),
                "shutdown": report.get("graceful_shutdown"),
            },
        )
        report.update(
            finished_utc=datetime.now(timezone.utc).isoformat(),
            summaries=summary,
            memory_analysis=analysis,
            correctness_checks=len(checks) + sum(r["correct"] is not None for r in requests),
            sampler_errors=sampler.errors if sampler else [],
            limitations="One traced local CPU process; no host isolation or confidence intervals. Tracemalloc starts after model load and cannot see most native memory. Idle diagnostics briefly acquire the work lock but never force GC. Completed run status is not a budget pass or leak-free certificate. Client timeout does not interrupt CPU inference; no cancellations are deliberately injected during steady soaks.",
        )
        try:
            if resources and requests:
                figures(output, resources, requests, [s[0] for s in stages])
        except Exception as exc:
            report.update(status="failed", plot_error=str(exc))
            raise
        finally:
            save(output / "phase6_1_report.json", report)
        print(
            json.dumps(
                {"status": report["status"], "artifact_directory": report["artifact_directory"]}
            ),
            flush=True,
        )
    save(ROOT / "experiments/reports" / (run_id + ".json"), report)


if __name__ == "__main__":
    main()
