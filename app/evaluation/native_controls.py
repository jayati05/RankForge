"""Reproducible isolated native controls in fresh processes, outside serving soaks."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("all", "blas", "inference"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "status": "running",
        "runs": [],
    }
    destination = args.output / "controls_manifest.json"

    def save():
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save()
    cases = []
    if args.mode in ("all", "blas"):
        cases += [
            (
                f"blas_{count}",
                "app.evaluation.native_control",
                [],
                {"OPENBLAS_NUM_THREADS": str(count)},
            )
            for count in (1, 20)
        ]
    if args.mode in ("all", "inference"):
        cases += [
            (method, "app.evaluation.native_operator", ["--method", method], {})
            for method in ("hybrid", "reranked")
        ]
        cases += [
            (
                "reranked_blas1",
                "app.evaluation.native_operator",
                ["--method", "reranked"],
                {"OPENBLAS_NUM_THREADS": "1"},
            )
        ]
    for label, module, extra, changes in cases:
        command = [sys.executable, "-m", module, *extra, "--output", str(args.output / label)]
        env = {k: v for k, v in os.environ.items() if not k.startswith("RANKFORGE_")}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1", **changes)
        row = {
            "label": label,
            "command": command,
            "environment_overrides": changes,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["runs"].append(row)
        save()
        with (args.output / (label + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        row.update(
            exit_code=result.returncode,
            status="complete" if result.returncode == 0 else "failed",
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        save()
    manifest.update(
        status="complete" if all(r["status"] == "complete" for r in manifest["runs"]) else "failed",
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    save()
    print(json.dumps({"status": manifest["status"], "output": str(args.output)}))
    if manifest["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
