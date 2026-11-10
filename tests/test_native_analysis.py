import os
import threading

import pytest

from app.evaluation.native_analysis import classification, latency, memory_summary, variability


def test_budget_failure_takes_priority_over_unknown():
    assert classification({}) == "Inconclusive"
    assert classification({"growth": None, "errors": False}) == "Failed"
    assert classification({"growth": None, "errors": True}) == "Inconclusive"
    assert classification({"growth": True, "errors": True}) == "Passed"


def test_rejection_latency_does_not_include_success_or_timeout():
    rows = [
        {"outcome": outcome, "latency_ms": ms, "status": status}
        for outcome, ms, status in [
            ("success", 900, 200),
            ("http_error", 2, 429),
            ("http_error", 4, 429),
            ("timeout", 30000, None),
        ]
    ]
    assert latency(rows, "rejected")["mean_ms"] == 3
    assert latency(rows, "success")["p95_ms"] == 900
    assert latency(rows, "cancelled")["p95_ms"] is None


def test_commit_trends_and_actual_idle_recovery_exclude_diagnostic_gap():
    rows = [
        {"stage": "soak", "elapsed_seconds": t, "private_commit_bytes": (100 + t / 60) * 2**20}
        for t in (0, 60, 120, 180, 240, 300)
    ]
    rows.extend(
        [
            {"stage": "soak_idle", "elapsed_seconds": 310, "private_commit_bytes": 120 * 2**20},
            {"stage": "soak_idle", "elapsed_seconds": 610, "private_commit_bytes": 118 * 2**20},
        ]
    )
    result = memory_summary(rows, "soak", "private_commit_bytes")
    assert result["whole_slope_mib_per_minute"] == pytest.approx(1)
    assert result["late_slope_mib_per_minute"] == pytest.approx(1)
    assert result["idle_only_delta_mib"] == -2
    assert result["late_trend_sufficient_duration"]


def test_short_memory_series_and_missing_series_not_stability_evidence():
    rows = [{"stage": "short", "elapsed_seconds": t, "rss_bytes": 42 * 2**20} for t in (0, 10)]
    assert not memory_summary(rows, "short", "rss_bytes")["late_trend_sufficient_duration"]
    assert memory_summary(rows, "missing", "rss_bytes") == {"status": "unavailable"}


def test_variability_retains_both_runs_and_unknown_single_sample_spread():
    result = variability([1, 3])
    assert result["n"] == 2
    assert result["min"] == 1 and result["max"] == 3 and result["mean"] == 2
    assert result["sample_standard_deviation"] == pytest.approx(2**0.5)
    assert variability([1])["sample_standard_deviation"] is None


def test_full_export_keeps_failures_and_sparse_native_observations(tmp_path, monkeypatch):
    """Authored local files: never download data or present fixtures as measurements."""
    import csv
    import json
    import sys
    from app.evaluation import native_analysis
    from app.evaluation.api_workload import aggregate, write_csv

    def save(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    campaign = {"status": "complete", "runs": []}
    for number, concurrency in enumerate((1, 2, 1, 2, 4)):
        run = tmp_path / str(number)
        run.mkdir()
        stage = "stress_c4" if concurrency == 4 else "soak"
        rows = [
            {
                "stage": stage,
                "method": "bm25",
                "outcome": outcome,
                "status": status,
                "latency_ms": ms,
            }
            for outcome, status, ms in (
                ("success", 200, 100),
                ("http_error", 429, 5),
                ("timeout", None, 30000),
            )
        ]
        stats = aggregate(rows, 300 if concurrency == 4 else 900)
        stats.update(
            stage=stage,
            concurrency=concurrency,
            stress=concurrency == 4,
            cancelled=0,
            submission_duration_seconds=300 if concurrency == 4 else 900,
        )
        stats["by_method"] = {"bm25": aggregate(rows, stats["elapsed_seconds"])}
        checks = {"timeouts": False, "late_memory_growth": None}
        report = {
            "status": "complete",
            "run_id": str(number),
            "process_ids": {"serving": number + 100},
            "config": {"warmup": 60, "idle": 300, "sample_interval": 1, "timeout": 30, "seed": 42},
            "workload_sha256": "authored-fixture",
            "summaries": [stats],
            "memory_analysis": [
                {"budgets": {"checks": checks, "other_errors": 0, "rejection_rate": 1 / 3}}
            ],
            "graceful_shutdown": "fixture",
            "correctness_checks": 1,
            "sampler_errors": [],
        }
        save(run / "phase6_1_report.json", report)
        write_csv(run / "request_outcomes.csv", rows)
        write_csv(
            run / "memory_timeseries.csv",
            [
                {
                    "stage": stage,
                    "elapsed_seconds": t,
                    "rss_bytes": 100 * 2**20,
                    "private_commit_bytes": 200 * 2**20,
                }
                for t in (0, 300)
            ],
        )
        snapshot = {
            "pid": number + 100,
            "label": "after_idle",
            "native": {
                "counters": {
                    "native_threads": 7,
                    "rss_bytes": 100 * 2**20,
                    "private_commit_bytes": 200 * 2**20,
                },
                "regions": {"totals": {"committed_bytes": 210 * 2**20}},
            },
        }
        save(run / "allocation_diagnostics.json", [snapshot])
        save(run / "command_1.json", {"command": "snapshot", "diagnostics": snapshot})
        campaign["runs"].append({"label": str(number), "artifact_directories": [str(run)]})
    save(tmp_path / "campaign.json", campaign)
    monkeypatch.setattr(sys, "argv", ["native_analysis", str(tmp_path)])
    monkeypatch.setattr(native_analysis, "figures", lambda *args: None)
    native_analysis.main()
    result = json.loads((tmp_path / "phase6_2_report.json").read_text())
    assert result["status"] == "complete" and result["full_duration_design_satisfied"]
    assert result["budget_classification"] == "Failed"
    assert all(run["budget_classification"] == "Failed" for run in result["runs"])
    assert result["runs"][0]["rejection_latency"]["p95_ms"] == 5
    assert result["runs"][0]["by_method"]["bm25"]["timed_out"] == 1
    assert result["runs"][0]["by_method"]["bm25"]["rejection_rate"] == pytest.approx(1 / 3)
    with (tmp_path / "request_outcomes.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 15
    with (tmp_path / "thread_count_timeseries.csv").open() as stream:
        native_rows = list(csv.DictReader(stream))
    assert len(native_rows) == 5
    assert all(r["native_threads"] == "7" for r in native_rows)
    assert native_rows[0]["elapsed_seconds"] == ""  # No invented1s timestamps.
    assert native_rows[0]["observation_scope"] == "lifecycle_checkpoint"


def test_profile_installs_sampler_in_executing_module(monkeypatch):
    """The CLI __main__ namespace must receive live native samples, not a dead copy."""
    from types import ModuleType
    import app.api.main as api
    from app.evaluation import api_soak, native_diagnostics, soak_diagnostics

    class AuthoredSampler:
        def __init__(self):
            self.pid, self.rows, self.errors = 123, [], []

        def _sample(self):
            self.rows.append({"rss_bytes": 100, "private_commit_bytes": 200})

    executing = ModuleType("__main__")
    executing.SoakSampler = AuthoredSampler
    canonical = api_soak.SoakSampler
    # Register original values for restoration after this benchmark-only hook.
    monkeypatch.setattr(api, "create_app", api.create_app)
    monkeypatch.setattr(soak_diagnostics, "Diagnostics", soak_diagnostics.Diagnostics)
    monkeypatch.setattr(native_diagnostics, "counters", lambda pid: {"native_threads": 7})
    monkeypatch.setattr(
        native_diagnostics,
        "regions",
        lambda pid, details=True: {"totals": {"virtual_allocated_bytes": 300}},
    )
    native_diagnostics.install(executing)
    assert executing.SoakSampler is not AuthoredSampler
    assert api_soak.SoakSampler is canonical
    sampler = executing.SoakSampler()
    sampler._sample()
    assert sampler.rows[0]["native_threads"] == 7
    assert sampler.rows[0]["virtual_allocated_bytes"] == 300
    assert sampler.rows[0]["native_sampling_ms"] >= 0
    assert not sampler.errors


@pytest.mark.skipif(os.name != "nt", reason="Windows native API integration")
def test_native_census_tracks_live_process_and_thread_lifecycle():
    from app.evaluation.windows_native import counters, regions

    pid = os.getpid()
    before = counters(pid)
    ready = threading.Event()
    stop = threading.Event()

    def worker():
        ready.set()
        stop.wait(10)

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(5)
    try:
        during = counters(pid)
        assert during["native_threads"] >= before["native_threads"] + 1
        assert during["rss_bytes"] > 0 and during["private_commit_bytes"] > 0
        result = regions(pid, details=False)
        totals = result["totals"]
        assert result["pid"] == pid
        assert (
            totals["virtual_allocated_bytes"]
            == totals["committed_bytes"] + totals["reserved_bytes"]
        )
        assert totals["committed_private_bytes"] > 0
    finally:
        stop.set()
        thread.join(5)
