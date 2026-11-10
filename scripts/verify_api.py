"""Real loopback HTTP verification against preserved Phase 2 SciFact rankings.

Run explicitly: python scripts/verify_api.py. Requires prepared local resources;
does not download, build indexes, calculate quality metrics or replace old runs.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api.schemas import CompareResponse, SearchResponse  # noqa: E402


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def previous_artifacts() -> dict[str, str]:
    paths = []
    for directory in ("experiments/results", "experiments/reports", "experiments/configs"):
        paths.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file()
            and "phase5" not in p.relative_to(ROOT).parts
            and not p.name.startswith("phase5_")
        )
    for directory in ("indexes", "models"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104",
    )
    parser.add_argument("--query-id", default="1241")
    parser.add_argument("--warm-requests", type=int, default=3)
    args = parser.parse_args()
    args.baseline = args.baseline.resolve()
    if not 1 <= args.warm_requests <= 20:
        parser.error("warm-requests must be from 1 to 20")
    old_hashes = previous_artifacts()
    run_id = (
        "phase5-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    )
    output = ROOT / "experiments/results/phase5" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save(output / "previous_artifacts_sha256.json", old_hashes)
    report = {
        "run_id": run_id,
        "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_directory": output.relative_to(ROOT).as_posix(),
        "query_id": args.query_id,
        "dataset": "BEIR SciFact",
        "document_count": sum(
            1
            for line in (ROOT / "data/public/scifact/corpus.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ),
        "top_k": 10,
        "baseline": args.baseline.relative_to(ROOT).as_posix(),
        "checks": [],
        "samples": [],
    }
    report["service_code_sha256"] = {
        path.relative_to(ROOT).as_posix(): digest(path)
        for directory in ("app/api", "app/core", "app/services", "scripts")
        for path in sorted((ROOT / directory).glob("*.py"))
    }
    child = None
    try:
        queries = [
            json.loads(line)
            for line in (ROOT / "data/public/scifact/queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        query = next(q["text"] for q in queries if q["_id"] == args.query_id)
        expected = json.loads((args.baseline / "rankings.json").read_text())
        protocol = json.loads(
            (ROOT / "experiments/configs/phase4-transfer/protocol.json").read_text()
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {k: v for k, v in os.environ.items() if not k.startswith("RANKFORGE_")}
        env.update(
            {
                "RANKFORGE_HOST": "127.0.0.1",
                "RANKFORGE_PORT": str(port),
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        with (output / "server.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, "-m", "app.api"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False
            ) as client:
                deadline = time.monotonic() + 30
                while True:
                    if child.poll() is not None:
                        raise RuntimeError("API process exited; inspect server.log")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("API startup timed out; inspect server.log")
                    threading.Event().wait(0.1)

                def check(condition, name):
                    if not condition:
                        raise AssertionError(name)
                    report["checks"].append(name)

                for route in ("/health", "/models", "/methods", "/docs", "/redoc", "/openapi.json"):
                    response = client.get(route)
                    check(
                        response.status_code == 200 and bool(response.headers.get("x-request-id")),
                        route + " HTTP 200 with request ID",
                    )
                    if route == "/openapi.json":
                        save(output / "openapi.json", response.json())
                cold = client.get("/health").json()
                check(
                    all(s["state"] == "not_loaded" for s in cold["resources"].values()),
                    "status endpoints initialize no resources",
                )
                save(output / "health_before.json", cold)
                responses = {}

                def verify_result(body):
                    method = body["method"]
                    baseline_method = "hybrid" if method == "router" else method
                    hits = expected[baseline_method][args.query_id][:10]
                    check(
                        [(r["document_id"], r["rank"]) for r in body["documents"]]
                        == [(r["document_id"], r["rank"]) for r in hits],
                        method + " exact preserved top-10 IDs/ranks",
                    )
                    check(
                        all(
                            abs(actual["score"] - saved["score"]) < 1e-5
                            for actual, saved in zip(body["documents"], hits)
                        ),
                        method + " preserved scores within 1e-5",
                    )
                    if method == "router":
                        check(
                            body["selected_strategy"] == "hybrid"
                            and not body["routing"]["reranker_invoked"],
                            "router uses saved hybrid fallback",
                        )
                        check(
                            body["routing"]["policy_record_sha256"]
                            == protocol["policy_record_sha256"],
                            "exact frozen policy identity",
                        )

                # Router precedes standalone dense to prove it initializes only the resources selected.
                for method in ("bm25", "router", "dense", "hybrid", "reranked"):
                    for index in range(args.warm_requests + 1):
                        start = time.perf_counter()
                        response = client.post(
                            "/search", json={"query": query, "method": method, "top_k": 10}
                        )
                        wall_ms = (time.perf_counter() - start) * 1000
                        check(response.status_code == 200, method + " search HTTP 200")
                        body = response.json()
                        SearchResponse.model_validate(body)
                        check(
                            body["request_id"] == response.headers["x-request-id"],
                            "response/log correlation header",
                        )
                        verify_result(body)
                        if index:
                            check(
                                body["latency"]["initialization_ms"] == 0
                                and not body["initialized_resources"],
                                method + " warm request reuses resources",
                            )
                        report["samples"].append(
                            {
                                "method": method,
                                "stage": "first_call" if index == 0 else "warm",
                                "http_round_trip_ms": wall_ms,
                                "http_processing_ms": float(response.headers["x-process-time-ms"]),
                                "latency": body["latency"],
                                "initialized_resources": body["initialized_resources"],
                            }
                        )
                        responses[method] = body
                    if method == "router":
                        check(
                            client.get("/models").json()["models"]["reranker"]["state"]
                            == "not_loaded",
                            "hybrid-only router does not load cross-encoder",
                        )
                compare = client.post(
                    "/compare",
                    json={
                        "query": query,
                        "methods": ["bm25", "dense", "hybrid", "reranked", "router"],
                        "top_k": 10,
                    },
                )
                check(compare.status_code == 200, "compare all five methods HTTP 200")
                body = compare.json()
                CompareResponse.model_validate(body)
                check(len(body["agreement"]) == 10, "ten pairwise rank agreements")
                for result in body["results"]:
                    verify_result(result)
                responses["compare"] = body
                for payload in (
                    {"query": ""},
                    {"query": "x", "method": "unknown"},
                    {"query": "x", "top_k": 21},
                    {"query": "x", "model_path": "private-marker"},
                ):
                    response = client.post("/search", json=payload)
                    check(
                        response.status_code == 422 and "private-marker" not in response.text,
                        "invalid request has safe HTTP 422",
                    )
                check(
                    client.post("/search", content=b"x" * 17000).status_code == 413,
                    "oversized body HTTP 413",
                )
                check(client.get("/missing").status_code == 404, "unknown route HTTP 404")
                check(client.get("/search").status_code == 405, "wrong verb HTTP 405")
                check(
                    client.post(
                        "/search", json={"query": "API_SENSITIVE_CANARY_DO_NOT_LOG"}
                    ).status_code
                    == 200,
                    "sensitive-query logging canary accepted",
                )
                health = client.get("/health").json()
                save(output / "health_after.json", health)
                save(output / "responses.json", responses)
                check(
                    all(s["state"] == "ready" for s in health["resources"].values()),
                    "all requested resources ready",
                )
                report["policy_record_sha256"] = protocol["policy_record_sha256"]
                report["models"] = client.get("/models").json()["models"]
                report["versions"] = {
                    p: version(p)
                    for p in (
                        "rankforge",
                        "fastapi",
                        "starlette",
                        "pydantic",
                        "uvicorn",
                        "httpx",
                        "anyio",
                        "numpy",
                        "sentence-transformers",
                        "transformers",
                        "torch",
                    )
                }
                report["python"] = sys.version
                report["platform"] = platform.platform()
                report["configuration"] = {
                    "device": "cpu",
                    "threads": 4,
                    "batch_size": 16,
                    "candidate_depth": 50,
                    "rrf_constant": 60,
                    "rerank_candidates": 20,
                    "workers": 1,
                    "read_only_cache": True,
                    "offline_models": True,
                }
            child.terminate()  # Stop only the child created above; unit tests separately exercise graceful lifespan shutdown.
            child.wait(timeout=30)
            log.flush()
        events = []
        text = (output / "server.log").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("{"):
                events.append(json.loads(line))
        counts = Counter(e["resource"] for e in events if e.get("event") == "resource_initialized")
        if any(counts.get(role) != 1 for role in ("encoder", "scorer")):
            raise AssertionError("Neural resources must initialize exactly once")
        if "API_SENSITIVE_CANARY_DO_NOT_LOG" in text or query in text:
            raise AssertionError("Full query appeared in server logs")
        report["initialization_event_counts"] = dict(counts)
        report["checks"].extend(
            ["encoder and scorer each initialized once", "full queries absent from server logs"]
        )
        changed = [
            name
            for name, sha in old_hashes.items()
            if not (ROOT / name).is_file() or digest(ROOT / name) != sha
        ]
        if changed:
            raise AssertionError("Previously existing artifacts changed: " + repr(changed))
        report["preserved_artifact_count"] = len(old_hashes)
        report["checks"].append("previous experiment/config/model/index bytes unchanged")
        report["latency_summary"] = []
        for method in ("bm25", "dense", "hybrid", "reranked", "router"):
            samples = [s for s in report["samples"] if s["method"] == method]
            warm = [s for s in samples if s["stage"] == "warm"]
            report["latency_summary"].append(
                {
                    "method": method,
                    "warm_request_count": len(warm),
                    "first_http_ms": samples[0]["http_round_trip_ms"],
                    "first_initialization_ms": samples[0]["latency"]["initialization_ms"],
                    "warm_mean_http_ms": mean(s["http_round_trip_ms"] for s in warm),
                    "warm_mean_retrieval_ms": mean(s["latency"]["retrieval_ms"] for s in warm),
                }
            )
        report["status"] = "complete"
        report["limitations"] = (
            "One public query, sequential requests, three warm requests per method by default, one CPU process. Shared first-use initialization follows bm25/router/dense/hybrid/reranked order. Not a load test, quality evaluation or representative latency benchmark; no percentiles/confidence intervals. HTTP wall time includes response receipt, excludes client response JSON decoding. Service timing excludes FastAPI JSON encoding/network; rerank_ms is included in retrieval_ms. Windows child is explicitly terminated after HTTP checks; graceful shutdown is exercised separately by lifespan tests."
        )
        report["checks"] = list(dict.fromkeys(report["checks"]))
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "verification.json", report)
        print(
            json.dumps(
                {"status": report["status"], "artifact_directory": report["artifact_directory"]}
            )
        )
    portable = ROOT / "experiments/reports" / (run_id + ".json")
    save(portable, report)
    print(json.dumps(report["latency_summary"], indent=2))


if __name__ == "__main__":
    main()
