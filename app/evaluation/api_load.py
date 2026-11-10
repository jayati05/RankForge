"""Explicit Phase 6 local HTTP load/resource verification; never reads qrels.

python -m app.evaluation.api_load --requests 100 --clients 1,2,4,8,16
Internal child roles use the same app factory/local retrievers; private files control
only this benchmark's child process, with no HTTP administration endpoint.
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx

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

ROOT = Path(__file__).resolve().parents[2]


def post_control(path: Path, sequence: int, command: str) -> None:
    temporary = path.with_suffix(".tmp")
    save(temporary, {"sequence": sequence, "command": command})
    temporary.replace(path)


def serving_pid(log_text: str) -> int:
    match = re.search(r"Started server process \[(\d+)\]", log_text)
    if not match:
        raise RuntimeError(
            "Cannot identify actual serving PID; refusing launcher memory measurements"
        )
    return int(match.group(1))


def save(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def serve(port: int, control_file: Path) -> None:
    import uvicorn
    from app.api.main import create_app

    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False))

    def control():
        previous = -1
        while not server.should_exit:
            message = json.loads(control_file.read_text(encoding="utf-8"))
            if message["sequence"] <= previous:
                threading.Event().wait(0.05)
                continue
            previous = message["sequence"]
            command = message["command"]
            if command.strip() == "initialize":
                manager = app.state.resources
                with manager.execution():
                    for name in METHODS:
                        manager.get(name)
            elif command.strip() == "stop":
                server.should_exit = True
                return

    threading.Thread(target=control, daemon=True).start()
    server.run()


def direct(output: Path, pool: list[dict]) -> None:
    from app.core.config import Settings
    from app.services.model_service import ModelService

    manager = ModelService(Settings())
    result = {}
    for method in METHODS:
        retriever = manager.get(method)
        result[method] = {
            q["id"]: [asdict(r) for r in retriever.search(q["text"], 10)] for q in pool
        }
    save(
        output,
        {
            "rankings": result,
            "document_count": len(manager.get("corpus")),
            "policy_record_sha256": manager.get("policy")[0]["policy_record_sha256"],
        },
    )
    manager.close()


def compare(hits: list[dict], expected: list[dict], tolerance: float = 1e-5) -> bool:
    return [(r["document_id"], r["rank"]) for r in hits] == [
        (r["document_id"], r["rank"]) for r in expected
    ] and all(abs(a["score"] - b["score"]) <= tolerance for a, b in zip(hits, expected))


async def batch(
    url: str, work: list[dict], clients: int, timeout: float, phase: str
) -> tuple[list[dict], float]:
    from app.api.schemas import SearchResponse

    rows = []
    async with httpx.AsyncClient(
        base_url=url,
        timeout=timeout,
        trust_env=False,
        limits=httpx.Limits(max_connections=clients, max_keepalive_connections=clients),
    ) as http:

        async def worker(number):
            for item in work[number::clients]:
                start = time.perf_counter()
                row = {
                    "phase": phase,
                    "concurrency": clients,
                    "request_index": item["request_index"],
                    "method": item["method"],
                    "query_id": item["query"]["id"],
                    "status": None,
                    "outcome": "transport_error",
                    "error_code": None,
                    "request_id": None,
                    "hits": None,
                }
                try:
                    response = await http.post(
                        "/search",
                        json={
                            "query": item["query"]["text"],
                            "method": item["method"],
                            "top_k": 10,
                        },
                    )
                    row["latency_ms"] = (time.perf_counter() - start) * 1000
                    row["status"] = response.status_code
                    body = response.json()
                    row["request_id"] = body.get("request_id")
                    if response.status_code == 200:
                        SearchResponse.model_validate(body)
                        row.update(
                            outcome="success",
                            hits=[
                                {k: r[k] for k in ("document_id", "rank", "score")}
                                for r in body["documents"]
                            ],
                            service_latency=body["latency"],
                            routing=body.get("routing"),
                        )
                    else:
                        row.update(
                            outcome="http_error", error_code=body.get("error", {}).get("code")
                        )
                except httpx.TimeoutException as exc:
                    row.update(outcome="timeout", error_code=type(exc).__name__)
                except (httpx.HTTPError, ValueError) as exc:
                    row.update(outcome="transport_error", error_code=type(exc).__name__)
                row.setdefault("latency_ms", (time.perf_counter() - start) * 1000)
                rows.append(row)

        start = time.perf_counter()
        await asyncio.gather(*(worker(number) for number in range(clients)))
        seconds = time.perf_counter() - start
    return sorted(rows, key=lambda r: r["request_index"]), seconds


def wait_idle(client: httpx.Client, seconds: float = 120) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        snapshot = client.get("/health").json()
        if snapshot["activity"]["active"] == 0 and snapshot["activity"]["queued"] == 0:
            return snapshot
        threading.Event().wait(0.02)
    raise RuntimeError("Server did not drain; no subsequent workload may overlap")


def figures(output: Path, summaries: list[dict], resources: list[dict]) -> None:
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
        ax.plot(
            [r["concurrency"] for r in summaries],
            [r["success_latency"][key] for r in summaries],
            marker="o",
            label=key,
        )
    ax.set(
        xlabel="Concurrent closed-loop clients",
        ylabel="Successful HTTP response latency (ms)",
        title="Phase 6 mixed workload; rejected requests reported separately",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "latency_vs_concurrency.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot([r["elapsed_seconds"] for r in resources], [r["rss_bytes"] / 2**20 for r in resources])
    ax.set(
        xlabel="Seconds since child monitor started",
        ylabel="Server working set / RSS (MiB)",
        title="Native process memory; 100ms samples plus stage markers",
    )
    fig.tight_layout()
    fig.savefig(output / "memory_usage.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", type=int, default=100, help="Measured requests per concurrency level"
    )
    parser.add_argument("--clients", default="1,2,4,8,16")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--initialization-timeout",
        type=float,
        default=900,
        help="Separate deadline for cold first-use and draining CPU work",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", default=",".join(m + ":1" for m in METHODS))
    parser.add_argument("--serve", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--control-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--direct", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        if not args.control_file:
            parser.error("Internal serve role requires its private control file")
        serve(args.serve, args.control_file)
        return
    pool = queries(ROOT / "data/public/scifact")
    if args.direct:
        direct(args.direct, pool)
        return
    levels = [int(n) for n in args.clients.split(",")]
    distribution = weights(args.methods)
    if (
        not 0 < args.initialization_timeout <= 3600
        or args.requests < 1
        or args.warmup < 0
        or not 0 < args.timeout <= 600
        or not levels
        or any(n < 1 or n > 64 for n in levels)
        or len(set(levels)) != len(levels)
    ):
        parser.error(
            "Positive requests, unique client levels 1..64, timeout (0,600], warmup >=0 required"
        )
    work = schedule(pool, args.requests, distribution, args.seed)
    run_id = (
        "phase6-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    )
    output = ROOT / "experiments/results/phase6" / run_id
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_directory": output.relative_to(ROOT).as_posix(),
        "command": subprocess.list2cmdline(
            [sys.executable, "-m", "app.evaluation.api_load", *sys.argv[1:]]
        ),
        "config": {
            "requests_per_level": args.requests,
            "concurrency_levels": levels,
            "request_timeout_seconds": args.timeout,
            "warmup_requests": args.warmup,
            "initialization_timeout_seconds": args.initialization_timeout,
            "seed": args.seed,
            "method_weights": distribution,
        },
        "workload_sha256": fingerprint(work),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p)
            for base in ("app/api", "app/services", "app/core", "app/evaluation")
            for p in sorted((ROOT / base).glob("*.py"))
        },
        "versions": {
            p: version(p)
            for p in (
                "fastapi",
                "starlette",
                "uvicorn",
                "httpx",
                "anyio",
                "numpy",
                "torch",
                "transformers",
                "sentence-transformers",
                "matplotlib",
            )
        },
    }
    save(output / "workload.json", {"queries": pool, "schedule": work})
    protected = {
        p.relative_to(ROOT).as_posix(): sha(p)
        for base in (
            "experiments/results",
            "experiments/reports",
            "experiments/configs",
            "models",
            "indexes",
        )
        for p in (ROOT / base).rglob("*")
        if p.is_file()
        and "phase6" not in p.relative_to(ROOT).parts
        and not p.name.startswith("phase6-")
    }
    save(output / "previous_artifacts_sha256.json", protected)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RANKFORGE_")}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    child = sampler = None
    control_file = output / "server_control.json"
    post_control(control_file, 0, "idle")
    summaries, rows, checks, probes = [], [], [], []
    try:
        # Isolate direct-model memory from the parent and measured API server.
        with (output / "direct.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "app.evaluation.api_load",
                    "--direct",
                    str(output / "direct_rankings.json"),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                creationflags=flags,
            )
        expected = json.loads((output / "direct_rankings.json").read_text())
        report["document_count"] = expected["document_count"]
        report["policy_record_sha256"] = expected["policy_record_sha256"]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with (output / "server.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "app.evaluation.api_load",
                    "--serve",
                    str(port),
                    "--control-file",
                    str(control_file),
                ],
                stdin=subprocess.DEVNULL,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            url = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=url, timeout=120, trust_env=False) as client:
                deadline = time.monotonic() + 30
                while True:
                    if child.poll() is not None:
                        raise RuntimeError("API child exited before startup")
                    try:
                        cold = client.get("/health").json()
                        break
                    except httpx.TransportError:
                        if time.monotonic() > deadline:
                            raise RuntimeError("API startup timed out")
                        threading.Event().wait(0.1)
                assert all(s["initialization_count"] == 0 for s in cold["resources"].values())
                pid = serving_pid((output / "server.log").read_text(encoding="utf-8"))
                report["process_ids"] = {"launcher": child.pid, "serving": pid}
                sampler = MemorySampler(pid)
                sampler.thread.start()
                sampler.mark("before_model_loading")
                save(output / "health_before.json", cold)
                # Simultaneous cold dense requests: rejection during long initialization is retained.
                cold_work = [
                    {"request_index": i, "method": "dense", "query": pool[i]} for i in range(4)
                ]
                cold_rows, cold_seconds = asyncio.run(
                    batch(url, cold_work, 4, args.initialization_timeout, "cold_initialization")
                )
                rows.extend(cold_rows)
                report["cold_initialization"] = aggregate(cold_rows, cold_seconds)
                state = wait_idle(client, args.initialization_timeout)
                assert state["resources"]["encoder"]["initialization_count"] == 1
                assert state["resources"]["dense"]["initialization_count"] == 1
                assert state["resources"]["scorer"]["initialization_count"] == 0
                routed = client.post(
                    "/search", json={"query": pool[0]["text"], "method": "router", "top_k": 10}
                )
                assert routed.status_code == 200
                assert (
                    routed.json()["routing"]["policy_record_sha256"]
                    == expected["policy_record_sha256"]
                )
                assert (
                    client.get("/models").json()["models"]["reranker"]["initialization_count"] == 0
                )
                post_control(control_file, 1, "initialize")
                deadline = time.monotonic() + args.initialization_timeout
                while True:
                    state = client.get("/health").json()
                    if all(v["state"] == "ready" for v in state["resources"].values()):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("Controlled resource initialization timed out")
                    threading.Event().wait(0.05)
                sampler.mark("after_model_initialization")
                warm = schedule(pool, max(1, args.warmup), distribution, args.seed)
                warm_rows = []
                if args.warmup:
                    warm_rows, _ = asyncio.run(batch(url, warm, 1, 120, "warmup"))
                    rows.extend(warm_rows)
                    assert all(r["outcome"] == "success" for r in warm_rows)
                sampler.mark("after_warmup")

                def correctness(stage):
                    for method in METHODS:
                        for query in pool:
                            response = client.post(
                                "/search",
                                json={"query": query["text"], "method": method, "top_k": 10},
                            )
                            ok = response.status_code == 200 and compare(
                                response.json()["documents"],
                                expected["rankings"][method][query["id"]],
                            )
                            checks.append(
                                {
                                    "stage": stage,
                                    "method": method,
                                    "query_id": query["id"],
                                    "passed": ok,
                                    "http_status": response.status_code,
                                }
                            )
                            if not ok:
                                raise AssertionError("Direct/API ranking or score mismatch")

                correctness("before_load")
                report["methods_differ_on_queries"] = [
                    q["id"]
                    for q in pool
                    if [r["document_id"] for r in expected["rankings"]["bm25"][q["id"]]]
                    != [r["document_id"] for r in expected["rankings"]["dense"][q["id"]]]
                ]
                for concurrency in levels:
                    sampler.mark(f"load_{concurrency}")
                    level_rows, seconds = asyncio.run(
                        batch(url, work, concurrency, args.timeout, "load")
                    )
                    rows.extend(level_rows)
                    summaries.append({"concurrency": concurrency, **aggregate(level_rows, seconds)})
                    for row in level_rows:
                        if row["outcome"] == "success":
                            ok = compare(
                                row["hits"], expected["rankings"][row["method"]][row["query_id"]]
                            )
                            if row["method"] == "router":
                                ok = (
                                    ok
                                    and row["routing"]["policy_record_sha256"]
                                    == expected["policy_record_sha256"]
                                )
                            checks.append(
                                {
                                    "stage": f"load_{concurrency}",
                                    "method": row["method"],
                                    "query_id": row["query_id"],
                                    "passed": ok,
                                }
                            )
                            if not ok:
                                raise AssertionError("Correctness changed under load")
                    wait_idle(client)
                    sampler.mark(f"after_load_{concurrency}")
                    print(
                        json.dumps(
                            {
                                "concurrency": concurrency,
                                **{
                                    k: summaries[-1][k]
                                    for k in ("successful", "failed", "statuses")
                                },
                            }
                        ),
                        flush=True,
                    )
                correctness("after_load")
                sampler.mark("after_requests_complete")
                # Client deadline: CPU work continues under its lock until it finishes.
                start = time.perf_counter()
                try:
                    client.post(
                        "/search",
                        json={"query": pool[0]["text"], "method": "reranked", "top_k": 10},
                        timeout=0.01,
                    )
                    probes.append(
                        {"probe": "client_timeout", "observed": "response_before_deadline"}
                    )
                except httpx.TimeoutException as exc:
                    probes.append(
                        {
                            "probe": "client_timeout",
                            "observed": type(exc).__name__,
                            "client_elapsed_ms": (time.perf_counter() - start) * 1000,
                            "activity_after_disconnect": client.get("/health").json()["activity"],
                        }
                    )
                wait_idle(client)

                async def cancelled():
                    async with httpx.AsyncClient(
                        base_url=url, timeout=120, trust_env=False
                    ) as http:
                        task = asyncio.create_task(
                            http.post(
                                "/search",
                                json={"query": pool[0]["text"], "method": "reranked", "top_k": 10},
                            )
                        )
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            if (await http.get("/health")).json()["activity"]["active"]:
                                break
                            await asyncio.sleep(0.01)
                        else:
                            raise RuntimeError("Cancellation probe never observed active request")
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            return {
                                "probe": "client_task_cancel",
                                "observed": "CancelledError",
                                "activity_after_cancel": (await http.get("/health")).json()[
                                    "activity"
                                ],
                            }
                        raise AssertionError("Client cancellation was not observed")

                probes.append(asyncio.run(cancelled()))
                wait_idle(client)
                recovery = client.post(
                    "/search", json={"query": pool[0]["text"], "method": "reranked", "top_k": 10}
                )
                assert recovery.status_code == 200 and compare(
                    recovery.json()["documents"], expected["rankings"]["reranked"][pool[0]["id"]]
                )
                report["post_cancellation_recovery_correct"] = True
                state = wait_idle(client)
                save(output / "health_after.json", state)
                report["initialization_counts"] = {
                    k: {n: v[n] for n in ("initialization_attempts", "initialization_count")}
                    for k, v in state["resources"].items()
                }
                assert all(v["initialization_count"] == 1 for v in state["resources"].values())
                # Verify real graceful shutdown during an active cross-encoder request.
                with httpx.Client(base_url=url, timeout=120, trust_env=False) as last:
                    result = {}

                    def pending():
                        try:
                            result["response"] = last.post(
                                "/search",
                                json={"query": pool[1]["text"], "method": "reranked", "top_k": 10},
                            )
                        except Exception as exc:
                            result["error"] = type(exc).__name__

                    thread = threading.Thread(target=pending)
                    thread.start()
                    deadline = time.monotonic() + 10
                    while not client.get("/health").json()["activity"]["active"]:
                        if time.monotonic() > deadline:
                            raise RuntimeError("Shutdown probe never became active")
                        threading.Event().wait(0.01)
                    post_control(control_file, 2, "stop")
                    thread.join(timeout=120)
                    assert (
                        not thread.is_alive()
                        and result.get("response") is not None
                        and result["response"].status_code == 200
                    )
                    assert compare(
                        result["response"].json()["documents"],
                        expected["rankings"]["reranked"][pool[1]["id"]],
                    )
                child.wait(timeout=30)
                assert child.returncode == 0
                report["graceful_shutdown_active_request"] = (
                    "HTTP 200, unchanged rankings, process exit 0"
                )
            log.flush()
        sampler.finish()
        required_stages = {
            "before_model_loading",
            "after_model_initialization",
            "after_warmup",
            "after_requests_complete",
        }
        if not required_stages <= {row["stage"] for row in sampler.rows}:
            raise RuntimeError("Required process memory stages are missing")
        events = [
            json.loads(line)
            for line in (output / "server.log").read_text(encoding="utf-8").splitlines()
            if line.startswith("{")
        ]
        report["initialization_log_counts"] = dict(
            Counter(e["resource"] for e in events if e.get("event") == "resource_initialized")
        )
        assert all(v == 1 for v in report["initialization_log_counts"].values())
        assert any(e.get("event") == "service_stopped" for e in events)
        report["protected_file_count"] = len(protected)
        assert all(
            (ROOT / p).is_file() and sha(ROOT / p) == value for p, value in protected.items()
        )
        report["protected_files_unchanged"] = True
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if sampler:
            sampler.finish()
        if child and child.poll() is None:
            try:
                post_control(control_file, 3, "stop")
                child.wait(timeout=120)
            except (OSError, subprocess.TimeoutExpired):
                child.terminate()
                child.wait(timeout=30)
                report["forced_child_cleanup"] = True
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        resources = sampler.rows if sampler else []
        save(output / "requests.json", rows)
        save(
            output / "load_test_summary.json",
            {
                "levels": summaries,
                "convention": "Closed-loop fixed requests per level; no retries. Throughput includes receipt of last client response, excludes warmup/correctness/probes. All-attempt and successful latency distributions separate; NumPy linear percentiles. Small tails are descriptive.",
            },
        )
        save(
            output / "correctness_checks.json",
            {
                "checks": checks,
                "all_passed": bool(checks) and all(c["passed"] for c in checks),
                "absolute_score_tolerance": 1e-5,
                "relative_score_tolerance": 0,
            },
        )
        save(
            output / "error_summary.json",
            {
                "non_success_requests": [r for r in rows if r["outcome"] != "success"],
                "probes": probes,
                "memory_sampler_errors": sampler.errors if sampler else [],
            },
        )
        if summaries:
            write_csv(
                output / "latency_by_concurrency.csv",
                [
                    {
                        **{
                            k: r[k]
                            for k in (
                                "concurrency",
                                "requests",
                                "successful",
                                "failed",
                                "timed_out",
                                "elapsed_seconds",
                                "attempts_per_second",
                                "successful_per_second",
                            )
                        },
                        **{"all_" + k: v for k, v in r["all_latency"].items()},
                        **{"success_" + k: v for k, v in r["success_latency"].items()},
                        "statuses": json.dumps(r["statuses"], sort_keys=True),
                    }
                    for r in summaries
                ],
            )
            method_rows = []
            for summary in summaries:
                for method in distribution:
                    subset = [
                        r
                        for r in rows
                        if r["phase"] == "load"
                        and r["concurrency"] == summary["concurrency"]
                        and r["method"] == method
                    ]
                    stats = aggregate(subset, summary["elapsed_seconds"])
                    method_rows.append(
                        {
                            "concurrency": summary["concurrency"],
                            "method": method,
                            **{
                                k: stats[k]
                                for k in ("requests", "successful", "failed", "timed_out")
                            },
                            **stats["success_latency"],
                            "statuses": json.dumps(stats["statuses"], sort_keys=True),
                        }
                    )
            write_csv(output / "method_latency.csv", method_rows)
        if resources:
            write_csv(output / "resource_usage.csv", resources)
            report["memory_stage_samples"] = {
                stage: next(r for r in resources if r["stage"] == stage)
                for stage in (
                    "before_model_loading",
                    "after_model_initialization",
                    "after_warmup",
                    "after_requests_complete",
                )
                if any(r["stage"] == stage for r in resources)
            }
            report["sampled_peak_rss_bytes"] = max(r["rss_bytes"] for r in resources)
            report["sampled_load_peak_rss_bytes"] = max(
                (r["rss_bytes"] for r in resources if r["stage"].startswith("load_")), default=None
            )
            report["os_lifetime_peak_rss_bytes"] = max(
                r["os_lifetime_peak_rss_bytes"] for r in resources
            )
        report["load_summary"] = summaries
        report["correctness_check_count"] = len(checks)
        report["limitations"] = (
            "Single local CPU process, one fixed seeded closed-loop pass per level; serialized inference, bounded queue, HTTP 429 retained. Percentiles are descriptive with small successful subsets. Working set/RSS sampled at 100ms may miss transients; OS peak is cumulative; private commit differs from RSS. Direct correctness worker exits before measured server starts. After-model-init stage includes earlier cold dense/router first-use inference, precedes configured warmup. Client cancellation/timeout does not interrupt CPU inference; serialization is retained until completion. No throughput capacity/production readiness, relevance-quality or universal correctness claim."
        )
        try:
            if summaries and resources:
                figures(output, summaries, resources)
        except Exception as exc:
            report.update(status="failed", plot_error=type(exc).__name__)
            raise
        finally:
            save(output / "phase6_report.json", report)
        print(json.dumps({"status": report["status"], "directory": str(output)}), flush=True)
    save(ROOT / "experiments/reports" / (run_id + ".json"), report)


if __name__ == "__main__":
    main()
