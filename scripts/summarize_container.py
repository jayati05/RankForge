"""Derive resource/lifecycle summaries from preserved Phase7A observations."""

import argparse
import ast
import csv
from collections import Counter
import json
from pathlib import Path


def object_field(value):
    if not value or value == "None":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Initial verifier CSV used Python dict repr; retain the original CSV.
        return ast.literal_eval(value)


def resource_summary(rows):
    result = {}
    for label in dict.fromkeys(row["label"] for row in rows):
        samples = [r for r in rows if r["label"] == label]

        def maximum(field):
            values = [float(r[field]) for r in samples if r.get(field)]
            return max(values) if values else None

        cpu = [r for r in samples if r.get("cpu.stat")]
        utilization = None
        if len(cpu) > 1:
            elapsed = float(cpu[-1]["elapsed_seconds"]) - float(cpu[0]["elapsed_seconds"])
            usage = int(object_field(cpu[-1]["cpu.stat"])["usage_usec"]) - int(
                object_field(cpu[0]["cpu.stat"])["usage_usec"]
            )
            utilization = 100 * usage / 1e6 / elapsed if elapsed > 0 else None
        result[label] = {
            "sample_count": len(samples),
            "sampled_process_peak_rss_bytes": maximum("process_rss_bytes"),
            "process_lifetime_peak_rss_bytes": maximum("process_peak_rss_bytes"),
            "sampled_private_commit_peak_bytes": maximum("private_commit_bytes"),
            "sampled_cgroup_peak_current_bytes": maximum("memory.current"),
            "cgroup_lifetime_peak_bytes": maximum("memory.peak"),
            "memory_limit_bytes": maximum("memory.max"),
            "cpu_max": next((r["cpu.max"] for r in samples if r.get("cpu.max")), None),
            "mean_cpu_percent_one_core_100": utilization,
            "cpu_observation": "Delta cgroup CPU use / observed wall time; includes health/observer processes; Linux VM quota scheduling, not host CPU use.",
            "cgroup_memory_events_last": object_field(samples[-1].get("memory.events")),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    report = json.loads((root / "phase7a_report.json").read_text())
    startup = json.loads((root / "container_startup_metrics.json").read_text())
    with (root / "container_resource_usage.csv").open(newline="", encoding="utf-8") as stream:
        resources = resource_summary(list(csv.DictReader(stream)))
    requests = json.loads((root / "request_results.json").read_text())
    outcomes = {}
    for label in dict.fromkeys(row["label"] for row in requests):
        selected = [row for row in requests if row["label"] == label]
        outcomes[label] = {
            "attempts": len(selected),
            "outcomes": dict(Counter(r["outcome"] for r in selected)),
            "statuses": dict(Counter(str(r["status"]) for r in selected)),
        }
    lifecycle = {}
    for label, values in startup.items():
        cold = values["cold_health"]["resources"]
        final = values.get("final_health", {}).get("resources", {})
        lifecycle[label] = {
            "cold_all_initialization_counts_zero": all(
                r["initialization_count"] == 0 for r in cold.values()
            ),
            "all_resources_initialized_once": bool(final)
            and all(
                r["initialization_count"] == 1 and r["initialization_attempts"] == 1
                for r in final.values()
            ),
            "sampler_errors": values.get("sampler_errors"),
        }
    measurements = {
        "run_id": report["run_id"],
        "resources": resources,
        "outcomes": outcomes,
        "lifecycle": lifecycle,
        "note": "Derived from unchanged raw observations. Completion/equivalence/budget readiness are separate. No growth classification for this short campaign.",
    }
    (root / "measurements.json").write_text(
        json.dumps(measurements, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
