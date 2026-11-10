"""Explicit real-MLflow integration verification; ordinary pytest uses client doubles."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from app.tracking.mlflow import TrackingConfig, client_for, register  # noqa: E402
from app.tracking.records import digest, load_record, metrics, parameters  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    run_id = (
        "phase7b-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    output = ROOT / "experiments/results/phase7b" / run_id
    output.mkdir(parents=True)
    report = {
        "run_id": run_id,
        "status": "running",
        "tracking_uri": args.tracking_uri,
        "commands": [],
        "imports": [],
        "metric_equality_checks": 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    cli = ["docker", "compose", "--profile", "tracking"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    server_stopped = False

    def command(parts, name=None, check=True):
        result = subprocess.run(
            parts,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        report["commands"].append({"command": parts, "exit_code": result.returncode})
        if name:
            (output / name).write_text(result.stdout + result.stderr, encoding="utf-8")
        if check and result.returncode:
            raise RuntimeError(f"Command failed; see {name}: {result.stderr[-700:]}")
        return result

    def equality(receipt, record):
        client = client_for(args.tracking_uri)
        parent_id = receipt["parent_run_id"]
        found = client.search_runs(
            [client.get_experiment_by_name("rankforge/" + record.family).experiment_id],
            filter_string=f"tags.`mlflow.parentRunId` = '{parent_id}'",
            max_results=1000,
        )
        found = sorted(found, key=lambda r: int(r.data.tags["rankforge.row_index"]))
        assert len(found) == len(record.rows)
        for actual, row in zip(found, record.rows, strict=True):
            expected = metrics(record, row)
            assert actual.info.status == "FINISHED"
            assert actual.data.params == parameters(record, row)
            assert actual.data.metrics == expected, (
                actual.info.run_id,
                expected,
                actual.data.metrics,
            )
            report["metric_equality_checks"] += len(expected)
        return [r.info.run_id for r in found]

    def artifact_hash(parent_id):
        with tempfile.TemporaryDirectory() as destination:
            path = client_for(args.tracking_uri).download_artifacts(
                parent_id, "source_manifest.json", destination
            )
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    try:
        command(
            cli + ["up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "mlflow"],
            "mlflow_start.log",
        )
        command(["docker", "version"], "docker_version.txt")
        with httpx.Client(base_url=args.tracking_uri, timeout=30, trust_env=False) as http:
            assert http.get("/health").status_code == 200
            ui = http.get("/")
            ui.raise_for_status()
            (output / "mlflow_ui.html").write_text(ui.text, encoding="utf-8")
            script = re.search(r'<script[^>]+src="([^"]+)"', ui.text)
            assert script and http.get(script[1]).status_code == 200
            report["ui_verification"] = {
                "html_status": 200,
                "javascript_asset_status": 200,
                "browser_rendering": "Not tested; HTTP HTML and JS asset verified.",
            }
            report["untrusted_host_status"] = http.get(
                "/", headers={"Host": "untrusted.invalid"}
            ).status_code
            assert report["untrusted_host_status"] == 403
        saved = [
            ROOT / "experiments/reports/phase2_scifact_full.json",
            ROOT / "experiments/reports/phase3_scifact.json",
            ROOT / "experiments/reports/phase4_nfcorpus.json",
            ROOT / "experiments/reports/phase6_1-20261109T162840Z-f2dea9f54ecb.json",
            ROOT / "experiments/reports/phase6_2-repeat-20261109T194018Z-c8b6ba59.json",
            ROOT / "experiments/reports/phase7a-20261110T063328Z-3ae104cc.json",
        ]
        # Guard all canonical report/input artifacts before registration.
        protected = set(saved)
        for source in saved:
            record = load_record(source)
            protected.update(record.artifacts)
        before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
        (output / "source_hashes_before.json").write_text(
            json.dumps(before, indent=2), encoding="utf-8"
        )
        config = TrackingConfig(True, args.tracking_uri, "rankforge", output / "receipts")
        for source in saved:
            record = load_record(source)
            result = register(source, config, historical=True, expected_sha256=digest(source))
            assert result["status"] in ("logged", "duplicate"), result
            children = equality(result, record)
            duplicate = register(source, config, historical=True)
            assert (
                duplicate["status"] == "duplicate"
                and duplicate["parent_run_id"] == result["parent_run_id"]
            )
            report["imports"].append(
                {
                    "source": str(source.relative_to(ROOT)),
                    "source_sha256": record.source_hash,
                    "parent_run_id": result["parent_run_id"],
                    "child_run_ids": children,
                    "duplicate_parent_reused": True,
                    "family": record.family,
                }
            )
            print(f"Verified {source.name}: {len(children)} rows", flush=True)
        new = command(
            [
                sys.executable,
                "-m",
                "app.evaluation.benchmark",
                "--dataset",
                "data/smoke",
                "--k",
                "10",
                "--output",
                str(output / "new_smoke"),
                "--tracking",
                "--tracking-uri",
                args.tracking_uri,
            ],
            "new_tracked_smoke.log",
        )
        response = next(
            json.loads(line)["tracking"]
            for line in new.stdout.splitlines()
            if line.startswith('{"tracking":')
        )
        assert response["status"] == "logged", response
        report["new_smoke"] = response
        equality(response, load_record(response["source"]))
        parent = report["imports"][0]["parent_run_id"]
        report["artifact_hash_before_restart"] = artifact_hash(parent)
        command(cli + ["restart", "mlflow"], "mlflow_restart.log")
        command(
            cli + ["up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "mlflow"],
            "mlflow_restart_health.log",
        )
        report["artifact_hash_after_restart"] = artifact_hash(parent)
        assert report["artifact_hash_before_restart"] == report["artifact_hash_after_restart"]
        old = command(cli + ["ps", "-q", "mlflow"]).stdout.strip()
        command(
            cli
            + [
                "up",
                "-d",
                "--no-build",
                "--force-recreate",
                "--wait",
                "--wait-timeout",
                "120",
                "mlflow",
            ],
            "mlflow_recreate.log",
        )
        new_id = command(cli + ["ps", "-q", "mlflow"]).stdout.strip()
        assert old != new_id
        report["recreated_different_container"] = True
        report["artifact_hash_after_recreation"] = artifact_hash(parent)
        assert report["artifact_hash_before_restart"] == report["artifact_hash_after_recreation"]
        for imported, source in zip(report["imports"], saved, strict=True):
            equality(imported, load_record(source))
        command(cli + ["stop", "--timeout", "30", "mlflow"], "mlflow_stopped.log")
        server_stopped = True
        # New tracked evaluation must succeed locally even while the server is stopped.
        unavailable = command(
            [
                sys.executable,
                "-m",
                "app.evaluation.benchmark",
                "--dataset",
                "data/smoke",
                "--output",
                str(output / "unavailable_smoke"),
                "--tracking",
                "--tracking-uri",
                args.tracking_uri,
            ],
            "unavailable_smoke.log",
        )
        failed = next(
            json.loads(line)["tracking"]
            for line in unavailable.stdout.splitlines()
            if line.startswith('{"tracking":')
        )
        assert (
            failed["status"] == "failed"
            and (Path(failed["source"]) / "benchmark_summary.csv").is_file()
        )
        report["unavailable_tracking_preserved_local_results"] = True
        command(
            cli + ["up", "-d", "--no-build", "--wait", "--wait-timeout", "90", "api"],
            "api_without_mlflow_start.log",
        )
        query = json.loads(
            (ROOT / "data/public/scifact/queries.jsonl").read_text().splitlines()[0]
        )["text"]
        # Use exact known workload query94 rather than assume input order.
        query = next(
            json.loads(line)["text"]
            for line in (ROOT / "data/public/scifact/queries.jsonl").read_text().splitlines()
            if json.loads(line)["_id"] == "94"
        )
        reference = json.loads(
            (
                ROOT
                / "experiments/results/phase7a/phase7a-20261110T063328Z-3ae104cc/local_rankings.json"
            ).read_text()
        )
        from app.evaluation.api_load import compare

        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=900, trust_env=False) as http:
            response = http.post(
                "/compare",
                json={
                    "query": query,
                    "methods": ["bm25", "dense", "hybrid", "reranked", "router"],
                    "top_k": 10,
                },
            )
            response.raise_for_status()
            report["api_without_mlflow"] = {}
            for result in response.json()["results"]:
                assert compare(result["documents"], reference[result["method"]]["94"])
                report["api_without_mlflow"][result["method"]] = True
                if result["method"] == "router":
                    assert (
                        result["routing"]["policy_record_sha256"]
                        == "31b0cef1d5969d2b79953be683050877038c34fabc4c560b4a1b8be8f90b67f8"
                    )
        command(cli + ["stop", "--timeout", "60", "api"], "api_without_mlflow_stop.log")
        after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
        assert before == after
        report["canonical_source_files_unchanged"] = len(after)
        command(
            cli + ["up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "mlflow"],
            "mlflow_recovered.log",
        )
        server_stopped = False
        report["artifact_hash_after_recovery"] = artifact_hash(parent)
        assert report["artifact_hash_before_restart"] == report["artifact_hash_after_recovery"]
        command(cli + ["config", "--format", "json"], "compose_configuration.json")
        command(cli + ["logs", "--no-color", "mlflow"], "mlflow_server.log")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if server_stopped:
            command(cli + ["up", "-d", "--no-build", "mlflow"], "cleanup_recover.log", check=False)
        command(cli + ["stop", "--timeout", "60", "api"], "cleanup_api.log", check=False)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        (output / "phase7b_report.json").write_text(text, encoding="utf-8")
        (ROOT / "experiments/reports" / (run_id + ".json")).write_text(text, encoding="utf-8")
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
