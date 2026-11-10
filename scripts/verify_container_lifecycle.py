"""Separate real-container active-shutdown and recreation probe; no ordinary tests."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import subprocess
import time
import uuid

import httpx

from verify_container import ROOT, port, wait_health


def main():
    name = "rf-lifecycle-" + uuid.uuid4().hex[:8]
    output = ROOT / "experiments/results/phase7a" / name
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        RANKFORGE_PUBLISHED_PORT=str(port()), RANKFORGE_MEMORY_LIMIT="4g", RANKFORGE_CPU_LIMIT="4"
    )
    report = {
        "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project": name,
        "commands": [],
    }
    prefix = ["docker", "compose", "-f", "compose.yaml", "-p", name]

    def run(parts, check=True):
        result = subprocess.run(
            parts,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        report["commands"].append(
            {
                "command": parts,
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
            }
        )
        if check and result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    def compose(*args, **kwargs):
        return run(prefix + list(args), **kwargs)

    def save():
        (output / "lifecycle.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        report["image_dependency_freeze"] = run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "rankforge:phase7a",
                "python",
                "-m",
                "pip",
                "freeze",
            ]
        )
        run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "rankforge:phase7a",
                "python",
                "-m",
                "pip",
                "check",
            ]
        )
        compose("up", "-d", "--no-build")
        cid = compose("ps", "-q", "api").strip()
        url = "http://127.0.0.1:" + env["RANKFORGE_PUBLISHED_PORT"]
        report["cold_health"], _ = wait_health(url)
        with httpx.Client(base_url=url, timeout=900, trust_env=False) as client:
            body = {"query": "vitamin D", "method": "reranked", "top_k": 10}
            client.post("/search", json=body).raise_for_status()
            routed = client.post("/search", json={**body, "method": "router"})
            routed.raise_for_status()
            report["policy_before"] = routed.json()["routing"]["policy_record_sha256"]
            report["healthcheck_exit_success"] = (
                run(["docker", "exec", cid, "python", "docker/healthcheck.py"]) == ""
            )
            report["initialized_health"] = client.get("/health").json()
            with ThreadPoolExecutor(max_workers=1) as pool:
                active = pool.submit(client.post, "/search", json=body)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if client.get("/health").json()["activity"]["active"] == 1:
                        break
                    time.sleep(0.01)
                else:
                    raise RuntimeError(
                        "Could not observe active inference; shutdown test is inconclusive"
                    )
                report["observed_active_before_stop"] = True
                compose("stop", "--timeout", "60")
                response = active.result(timeout=60)
                report["active_request_http_status"] = response.status_code
            state = json.loads(run(["docker", "inspect", cid]))[0]
            report["stopped_state"] = state["State"]
            log = compose("logs", "--no-color")
            (output / "server.log").write_text(log, encoding="utf-8")
            report["graceful"] = (
                response.status_code == 200
                and state["State"]["ExitCode"] == 0
                and not state["State"]["OOMKilled"]
                and "resources_released" in log
            )
            compose("up", "-d", "--no-build", "--force-recreate")
            report["recreated_health"], _ = wait_health(url)
            new_id = compose("ps", "-q", "api").strip()
            report["different_container_id"] = new_id != cid
            restored = client.post("/search", json={**body, "method": "router"})
            restored.raise_for_status()
            actual = restored.json()
            report["policy_after"] = actual["routing"]["policy_record_sha256"]
            report["recreated_rankings_equal"] = actual["documents"] == routed.json()["documents"]
            report["recreated_initialized_health"] = client.get("/health").json()
            report["lazy_scorer_after_recreation"] = (
                report["recreated_initialized_health"]["resources"]["scorer"][
                    "initialization_count"
                ]
                == 0
            )
            report["status"] = (
                "complete"
                if all(
                    report[k]
                    for k in (
                        "graceful",
                        "different_container_id",
                        "recreated_rankings_equal",
                        "lazy_scorer_after_recreation",
                    )
                )
                and report["policy_before"] == report["policy_after"]
                else "failed"
            )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        compose("down", "--timeout", "60", check=False)
        save()
        print(json.dumps({"status": report["status"], "directory": str(output)}), flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
