"""Explicit local Docker integration stage; ordinary pytest never invokes this.

Uses only its unique Compose project. Partial evidence survives every failure.
"""

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.evaluation.api_load import batch, compare, post_control, save, serving_pid, sha  # noqa: E402
from app.evaluation.api_workload import (  # noqa: E402
    METHODS,
    aggregate,
    fingerprint,
    memory,
    queries,
    schedule,
)


def port():
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return stream.getsockname()[1]


def csv_file(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            for row in rows
        )


def wait_health(url, seconds=90):
    start = time.perf_counter()
    with httpx.Client(base_url=url, timeout=3, trust_env=False) as client:
        while time.perf_counter() - start < seconds:
            try:
                response = client.get("/health")
                if response.status_code == 200:
                    return response.json(), time.perf_counter() - start
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
    raise RuntimeError("Health endpoint did not start within deadline")


def check_result(actual, expected, policy):
    ordered = [(d["document_id"], d["rank"]) for d in actual["documents"]] == [
        (d["document_id"], d["rank"]) for d in expected
    ]
    return {
        "ordered_ids_ranks_equal": ordered,
        "scores_within_1e_5": compare(actual["documents"], expected),
        "max_abs_score_difference": max(
            (abs(a["score"] - b["score"]) for a, b in zip(actual["documents"], expected)),
            default=None,
        )
        if ordered
        else None,
        "frozen_policy_equal": actual["routing"]["policy_record_sha256"] == policy
        if actual["method"] == "router"
        else True,
    }


def plots(output, summary, resources, startup):
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for label in dict.fromkeys(r["label"] for r in summary):
        subset = [r for r in summary if r["label"] == label and r["clients"] == 1]
        ax.plot(
            [r["method"] for r in subset],
            [r["success_latency"]["mean_ms"] for r in subset],
            marker="o",
            label=label,
        )
    ax.set(ylabel="Successful HTTP mean ms (C1)", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "api_latency.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    for label in dict.fromkeys(r["label"] for r in resources):
        subset = [r for r in resources if r["label"] == label and "process_rss_bytes" in r]
        ax.plot(
            [r["elapsed_seconds"] for r in subset],
            [r["process_rss_bytes"] / 2**20 for r in subset],
            label=label,
        )
    ax.set(xlabel="Seconds within run", ylabel="Process RSS MiB (not cgroup total)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "memory_usage.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    labels = list(startup)
    ax.bar(labels, [startup[x]["health_seconds"] for x in labels], label="Health")
    ax.bar(
        labels,
        [startup[x].get("first_use_seconds", 0) for x in labels],
        bottom=[startup[x]["health_seconds"] for x in labels],
        label="First use, all five (includes inference)",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.set(ylabel="Seconds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "startup_initialization.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--clients", default="1,2")
    parser.add_argument(
        "--limits", default="4g:4,2g:2", help="memory:cpus pairs, sequential fresh containers"
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        help="Explicit additional BLAS=thread experiment, never a default",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    levels = [int(x) for x in args.clients.split(",")]
    if args.requests < 1 or any(x < 1 for x in levels):
        parser.error("Positive requests/client counts required")
    run_id = (
        "phase7a-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    output = args.output or ROOT / "experiments/results/phase7a" / run_id
    output.mkdir(parents=True, exist_ok=False)
    project = "rf-" + run_id.lower()
    env = os.environ.copy()
    env.pop("RANKFORGE_EXPERIMENT_BLAS_THREADS", None)
    env["RANKFORGE_PUBLISHED_PORT"] = str(port())
    commands, rows, checks, resources, summaries, startup = [], [], [], [], [], {}
    report = {
        "run_id": run_id,
        "status": "running",
        "platform": platform.platform(),
        "python": sys.version,
        "command": [sys.executable, *sys.argv],
        "project": project,
        "limits": args.limits,
        "score_tolerance": 1e-5,
        "resource_observer": "~2s plus Docker exec duration; process RSS and cgroup-v2 total/peak separately",
        "host_container_process_memory": "Docker VM Linux PID is not a Windows PID; host process memory unavailable for that PID",
        "readiness": "Not established; short packaging experiment cannot resolve Phase 6.2 long-run budget failures.",
    }
    build_report = {"status": "not_run"}
    local = None
    local_log = None
    cid = None

    def command(parts, *, check=True, timeout=900, filename=None, command_env=None):
        start = time.perf_counter()
        result = subprocess.run(
            parts,
            cwd=ROOT,
            env=command_env or env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        entry = {
            "command": parts,
            "exit_code": result.returncode,
            "seconds": time.perf_counter() - start,
        }
        commands.append(entry)
        if filename:
            (output / filename).write_text(result.stdout, encoding="utf-8")
        if check and result.returncode:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {parts}; {result.stdout[-1500:]}"
            )
        return result

    def compose(*parts, **kwargs):
        files = ["-f", "compose.yaml"]
        if env.get("RANKFORGE_EXPERIMENT_BLAS_THREADS"):
            files += ["-f", "compose.blas-experiment.yaml"]
        return command(["docker", "compose", "-p", project, *files, *parts], **kwargs)

    def inspect():
        return json.loads(command(["docker", "inspect", cid]).stdout)[0]

    def exercise(label, url, reference, policy, pid=None):
        print(f"Starting {label}", flush=True)
        start = time.perf_counter()
        health, health_seconds = wait_health(url)
        startup[label] = {"health_seconds": health_seconds, "cold_health": health}
        stop = threading.Event()
        sampler_errors = []

        def sample():
            while not stop.is_set():
                try:
                    if pid:
                        value = memory(pid)
                        value = {
                            "process_rss_bytes": value["rss_bytes"],
                            "private_commit_bytes": value.get("private_commit_bytes"),
                        }
                    else:
                        value = json.loads(
                            command(
                                ["docker", "exec", cid, "python", "docker/observe.py"], timeout=10
                            ).stdout
                        )
                    resources.append(
                        {"label": label, "elapsed_seconds": time.perf_counter() - start, **value}
                    )
                except Exception as exc:
                    sampler_errors.append(type(exc).__name__)
                stop.wait(2)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            with httpx.Client(base_url=url, timeout=900, trust_env=False) as client:
                for endpoint in ("/methods", "/models", "/openapi.json", "/docs"):
                    response = client.get(endpoint)
                    if response.status_code != 200:
                        raise RuntimeError(f"{endpoint}: HTTP {response.status_code}")
                first = time.perf_counter()
                initializations = {}
                for method in METHODS:
                    response = client.post(
                        "/search", json={"query": pool[0]["text"], "method": method, "top_k": 10}
                    )
                    response.raise_for_status()
                    initializations[method] = response.json()["latency"]
                startup[label]["first_use_seconds"] = time.perf_counter() - first
                startup[label]["first_use_latency"] = initializations
                warm = time.perf_counter()
                for method in METHODS:
                    client.post(
                        "/search", json={"query": pool[1]["text"], "method": method, "top_k": 10}
                    ).raise_for_status()
                startup[label]["warmup_seconds"] = time.perf_counter() - warm
                all_refs = {}
                for method in METHODS:
                    all_refs[method] = {}
                    for q in pool:
                        response = client.post(
                            "/search", json={"query": q["text"], "method": method, "top_k": 10}
                        )
                        response.raise_for_status()
                        actual = response.json()
                        all_refs[method][q["id"]] = actual["documents"]
                        checks.append(
                            {
                                "label": label,
                                "stage": "reference",
                                "method": method,
                                "query_id": q["id"],
                                **check_result(actual, reference[method][q["id"]], policy),
                            }
                        )
                save(output / f"{label}_rankings.json", all_refs)
                compared = client.post(
                    "/compare",
                    json={"query": pool[0]["text"], "methods": list(METHODS), "top_k": 10},
                )
                compared.raise_for_status()
                for actual in compared.json()["results"]:
                    checks.append(
                        {
                            "label": label,
                            "stage": "compare_endpoint",
                            "method": actual["method"],
                            **check_result(
                                actual, reference[actual["method"]][pool[0]["id"]], policy
                            ),
                        }
                    )
                for clients in levels:
                    measured, seconds = asyncio.run(batch(url, work, clients, 30, label))
                    for row in measured:
                        row["label"] = label
                        if row["outcome"] == "success":
                            checks.append(
                                {
                                    "label": label,
                                    "stage": "workload",
                                    "method": row["method"],
                                    "ordered_ids_ranks_equal": [
                                        (x["document_id"], x["rank"]) for x in row["hits"]
                                    ]
                                    == [
                                        (x["document_id"], x["rank"])
                                        for x in reference[row["method"]][row["query_id"]]
                                    ],
                                    "scores_within_1e_5": compare(
                                        row["hits"], reference[row["method"]][row["query_id"]]
                                    ),
                                    "frozen_policy_equal": (row.get("routing") or {}).get(
                                        "policy_record_sha256"
                                    )
                                    == policy
                                    if row["method"] == "router"
                                    else True,
                                }
                            )
                    rows.extend(measured)
                    for method in METHODS:
                        subset = [r for r in measured if r["method"] == method]
                        summaries.append(
                            {
                                "label": label,
                                "clients": clients,
                                "method": method,
                                **aggregate(subset, seconds),
                            }
                        )
                startup[label]["final_health"] = client.get("/health").json()
                startup[label]["sampler_errors"] = sampler_errors
        finally:
            stop.set()
            thread.join(15)
        return all_refs

    try:
        command(["docker", "version"], filename="docker_version.txt")
        info = json.loads(
            command(
                ["docker", "info", "--format", "{{json .}}"], filename="docker_info.json"
            ).stdout
        )
        report["docker_memory_bytes"] = info["MemTotal"]
        save(
            output / "compose_configuration.json",
            json.loads(compose("config", "--format", "json").stdout),
        )
        if not args.skip_build:
            build = compose(
                "--progress", "plain", "build", filename="build.log", check=False, timeout=1800
            )
            build_report = {
                **commands[-1],
                "status": "complete" if build.returncode == 0 else "failed",
                "cache": "BuildKit cache allowed; see build.log",
            }
            if build.returncode:
                raise RuntimeError("Docker build failed; see build.log")
        else:
            build_report = {"status": "skipped_explicitly", "build_seconds": None}
        image = json.loads(
            command(
                ["docker", "image", "inspect", "rankforge:phase7a"], filename="image.json"
            ).stdout
        )[0]
        build_report.update(
            image_id=image["Id"],
            image_size_bytes=image["Size"],
            repo_digests=image["RepoDigests"],
            tag="rankforge:phase7a",
        )
        # Build Linux cache offline using existing indexing CLI. Never rewrite Windows cache.
        index = ROOT / "indexes/scifact-minilm-linux.npz"
        preparation = compose(
            "run",
            "--rm",
            "--no-deps",
            "--volume",
            f"{ROOT / 'indexes'}:/prepared",
            "api",
            "python",
            "-m",
            "app.retrieval.dense",
            "--dataset",
            "/resources/dataset",
            "--model-path",
            "/resources/models/dense",
            "--cache",
            "/prepared/scifact-minilm-linux.npz",
            filename="linux_index.log",
        )
        report["linux_index"] = {
            "sha256": sha(index),
            "command": commands[-1],
            "output": preparation.stdout,
        }
        protected = [
            p
            for root in (
                "models",
                "indexes",
                "experiments/configs/phase4-transfer",
                "data/public/scifact",
            )
            for p in (ROOT / root).rglob("*")
            if p.is_file()
        ]
        before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
        save(output / "resources_before.json", before)
        pool = queries(ROOT / "data/public/scifact")
        work = schedule(pool, args.requests, dict.fromkeys(METHODS, 1), 42)
        save(
            output / "workload.json",
            {"queries": pool, "schedule": work, "sha256": fingerprint(work)},
        )
        command(
            [
                sys.executable,
                "-m",
                "app.evaluation.api_load",
                "--direct",
                str(output / "direct.json"),
            ],
            filename="direct.log",
        )
        direct = json.loads((output / "direct.json").read_text())
        reference, policy = direct["rankings"], direct["policy_record_sha256"]
        local_port = port()
        control = output / "local_control.json"
        post_control(control, 0, "idle")
        local_log = (output / "local_server.log").open("w", encoding="utf-8")
        local_env = {k: v for k, v in os.environ.items() if not k.startswith("RANKFORGE_")}
        local_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        local = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "app.evaluation.api_load",
                "--serve",
                str(local_port),
                "--control-file",
                str(control),
            ],
            cwd=ROOT,
            env=local_env,
            stdin=subprocess.DEVNULL,
            stdout=local_log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        _, local_health_time = wait_health(f"http://127.0.0.1:{local_port}")
        pid = serving_pid((output / "local_server.log").read_text())
        exercise("local", f"http://127.0.0.1:{local_port}", reference, policy, pid)
        startup["local"]["health_seconds"] = local_health_time
        startup["local"]["launch_to_health_seconds"] = local_health_time
        post_control(control, 1, "stop")
        local.wait(60)
        report["local_graceful_shutdown"] = local.returncode == 0
        local_log.close()
        local_log = None
        local = None
        # A missing resource must fail the container entrypoint, not silently download.
        missing = compose(
            "run",
            "--rm",
            "--no-deps",
            "-e",
            "RANKFORGE_DATASET_DIR=/missing",
            "api",
            check=False,
            filename="missing_resource.log",
        )
        report["missing_resource_exit_code"] = missing.returncode
        cases = [(m, c, None) for m, c in (x.split(":") for x in args.limits.split(","))]
        if args.blas_threads:
            cases.append((*cases[0][:2], args.blas_threads))
        report["containers"] = []
        for i, (mem, cpu, blas) in enumerate(cases):
            label = f"container_{i}_{mem}_{cpu}" + (f"_blas{blas}" if blas else "")
            env.update(RANKFORGE_MEMORY_LIMIT=mem, RANKFORGE_CPU_LIMIT=cpu)
            if blas:
                env["RANKFORGE_EXPERIMENT_BLAS_THREADS"] = str(blas)
            started = time.perf_counter()
            compose("up", "-d", "--no-build", "--force-recreate", filename=f"{label}_up.log")
            cid = compose("ps", "-q", "api").stdout.strip()
            state = {
                "label": label,
                "memory_limit": mem,
                "cpu_limit": cpu,
                "experimental_blas_threads": blas,
                "container_id": cid,
            }
            report["containers"].append(state)
            url = "http://127.0.0.1:" + env["RANKFORGE_PUBLISHED_PORT"]
            try:
                _, startup_seconds = wait_health(url)
                state["launch_to_health_seconds"] = time.perf_counter() - started
                exercise(label, url, reference, policy)
                startup[label]["health_seconds"] = state["launch_to_health_seconds"]
                state["before_stop"] = inspect()
                compose("stop", "--timeout", "60")
                state["after_stop"] = inspect()
                compose("logs", "--no-color", filename=f"{label}_server.log")
                state["graceful"] = (
                    state["after_stop"]["State"]["ExitCode"] == 0
                    and "resources_released" in (output / f"{label}_server.log").read_text()
                )
                # Restart same container, verify cache counters reset then all forty query/method pairs.
                compose("start")
                exercise(label + "_restart", url, reference, policy)
                state["after_restart"] = inspect()
                compose("stop", "--timeout", "60")
                state["restart_stop"] = inspect()
                state["status"] = "complete"
            except Exception as exc:
                state.update(status="failed", error=str(exc), final_inspect=inspect())
                compose("logs", "--no-color", filename=f"{label}_failed.log", check=False)
            finally:
                compose("down", "--timeout", "60", check=False)
                cid = None
        after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
        save(output / "resources_after.json", after)
        report["persistent_resource_hashes_unchanged"] = before == after
        report["status"] = (
            "complete"
            if before == after
            and missing.returncode != 0
            and all(c["status"] == "complete" and c.get("graceful") for c in report["containers"])
            else "failed"
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if local:
            post_control(output / "local_control.json", 99, "stop")
            try:
                local.wait(60)
            except subprocess.TimeoutExpired:
                report["local_cleanup_failed"] = True
        if local_log:
            local_log.close()
        try:
            compose("down", "--timeout", "60", check=False)
        except Exception as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        correctness = {
            "checks": checks,
            "count": len(checks),
            "failed": sum(
                not all(
                    c[k]
                    for k in (
                        "ordered_ids_ranks_equal",
                        "scores_within_1e_5",
                        "frozen_policy_equal",
                    )
                )
                for c in checks
            ),
        }
        if correctness["failed"]:
            report["status"] = "failed"
        report.update(
            correctness_count=correctness["count"],
            correctness_failed=correctness["failed"],
            summaries=summaries,
            limitations="Short closed-loop workload; no long-run leak/boundedness conclusion. Linux CPU wheel/cache differs from Windows. Cgroup total includes page cache and exec/health observers; process RSS excludes those. Percentiles descriptive; no CPU isolation or equivalence claim without checks.",
        )
        save(output / "docker_build_report.json", build_report)
        save(output / "container_startup_metrics.json", startup)
        save(output / "container_correctness.json", correctness)
        save(
            output / "container_configuration.json",
            {
                "commands": commands,
                "containers": report.get("containers", []),
                "source_sha256": {
                    str(p.relative_to(ROOT)): sha(p)
                    for folder in ("app", "docker")
                    for p in (ROOT / folder).rglob("*.py")
                },
            },
        )
        save(output / "request_results.json", rows)
        csv_file(
            output / "container_resource_usage.csv",
            resources,
            [
                "label",
                "elapsed_seconds",
                "process_rss_bytes",
                "private_commit_bytes",
                "process_peak_rss_bytes",
                "process_threads",
                "memory.current",
                "memory.peak",
                "memory.max",
                "cpu.max",
                "cpu.stat",
                "memory.events",
            ],
        )
        csv_file(
            output / "container_api_latency.csv",
            rows,
            [
                "label",
                "phase",
                "concurrency",
                "request_index",
                "method",
                "query_id",
                "status",
                "outcome",
                "error_code",
                "latency_ms",
            ],
        )
        if summaries and resources:
            try:
                plots(output, summaries, resources, startup)
            except Exception as exc:
                report.update(status="failed", plot_error=f"{type(exc).__name__}: {exc}")
        save(output / "phase7a_report.json", report)
        save(ROOT / "experiments/reports" / (run_id + ".json"), report)
        print(
            json.dumps(
                {"status": report["status"], "directory": str(output), "error": report.get("error")}
            ),
            flush=True,
        )
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
