"""Independent Phase6.2 campaign; reuses the Phase6.1 workload and harness."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    base = root / "experiments/results/phase6_2"
    run_id = (
        "repeat-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    out = base / run_id
    out.mkdir(parents=True)
    report = {
        "campaign_id": run_id,
        "status": "running",
        "smoke": args.smoke,
        "runs": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "design": "Sequential independent serving processes; C1 twice/C2 twice/C4 once. Phase6.1 equal-five-method seeded200-request cycle and eight-query pool. C4 has stress rejection allowance, no admission-limit changes.",
    }
    path = out / "campaign.json"

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    print(str(out), flush=True)
    for label, clients, stress in [
        ("c1_r1", "1", False),
        ("c2_r1", "2", False),
        ("c1_r2", "1", False),
        ("c2_r2", "2", False),
        ("c4_stress", "", True),
    ]:
        before = set(base.glob("phase6_2-*"))
        command = [
            sys.executable,
            "-m",
            "app.evaluation.api_soak",
            "--native-profile",
            "--clients",
            clients,
            "--duration",
            str(10 if args.smoke else 900),
            "--stress-duration",
            str((10 if args.smoke else 300) if stress else 0),
            "--warmup",
            str(5 if args.smoke else 60),
            "--idle",
            str(5 if args.smoke else 300),
            "--sample-interval",
            "1",
            "--timeout",
            "30",
            "--seed",
            "42",
            "--window",
            "120",
        ]
        item = {
            "label": label,
            "command": command,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        report["runs"].append(item)
        save()
        with (out / (label + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            item["controller_pid"] = process.pid
            save()
            result = process.wait()
        created = set(base.glob("phase6_2-*")) - before
        item.update(
            exit_code=result,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            status="complete" if result == 0 else "failed",
            artifact_directories=[p.relative_to(root).as_posix() for p in sorted(created)],
        )
        save()
        print(json.dumps(item), flush=True)
    report.update(
        status="complete" if all(r["status"] == "complete" for r in report["runs"]) else "failed",
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    save()


if __name__ == "__main__":
    main()
