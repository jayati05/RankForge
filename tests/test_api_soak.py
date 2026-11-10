"""Authored diagnostic fixtures; no public model loads or long waits."""

import asyncio
import json

import pytest
import test_api_reliability as fixtures

from app.evaluation.api_soak import duration_batch, query_pool
from app.evaluation.soak_analysis import analyze, budgets, slope
from app.evaluation.soak_diagnostics import object_counts, system_memory

setup = fixtures.setup
live = fixtures.live
isolated_log_handlers = fixtures.isolated_log_handlers


def samples(stage, seconds, rate):
    return [
        {"stage": stage, "elapsed_seconds": float(t), "rss_bytes": (100 + rate * t / 60) * 2**20}
        for t in range(seconds + 1)
    ]


def test_linear_growth_rate_and_explicit_budget_violation():
    rows = samples("soak", 600, 3)
    result = analyze(rows, "soak", 120, 2)
    assert slope(rows) == pytest.approx(3)
    assert result["late_slope_mib_per_minute"] == pytest.approx(3)
    assert result["classification"] == "increasing"
    assert result["growth_budget_passed"] is False


def test_late_plateau_and_idle_recovery_do_not_prove_a_leak():
    rows = samples("soak", 600, 0)
    for row in rows:
        row["rss_bytes"] = (100 + min(row["elapsed_seconds"], 200) / 10) * 2**20
    rows += [
        {"stage": "soak_idle", "elapsed_seconds": 601, "rss_bytes": 110 * 2**20},
        {"stage": "soak_recovery_checks", "elapsed_seconds": 602, "rss_bytes": 999 * 2**20},
    ]
    result = analyze(rows, "soak", 120, 2)
    assert result["classification"] == "approximately_flat_late_window"
    assert result["whole_slope_mib_per_minute"] > 0
    assert result["idle_end_mib"] == 110
    assert result["idle_change_mib"] == -10
    assert "do not prove" in result["interpretation"]


def test_short_windows_are_indeterminate_not_stability_success():
    result = analyze(samples("soak", 10, 0), "soak", 120, 2)
    assert result["classification"] == "insufficient_duration"
    assert result["growth_budget_passed"] is None
    assert slope([]) is None


def test_rejection_and_latency_budgets_use_successful_distribution():
    stats = {
        "requests": 100,
        "failed": 10,
        "timed_out": 0,
        "cancelled": 0,
        "statuses": {"200": 90, "429": 10},
        "successful": 90,
        "success_latency": {"p95_ms": 2000},
    }
    memory = {"peak_mib": 1000, "growth_budget_passed": True}
    normal = budgets(stats, memory, 2048, 0.05, 5000)
    stress = budgets(stats, memory, 2048, 1, 5000)
    assert normal["checks"]["rejection_rate"] is False
    assert normal["rejection_rate"] == 0.1
    assert stress["all_passed"] is True
    assert normal["other_errors"] == 0


def test_timeout_and_cancellation_counts_are_not_hidden():
    stats = {
        "requests": 10,
        "failed": 3,
        "timed_out": 1,
        "cancelled": 1,
        "statuses": {"200": 7, "timeout": 1, "cancelled": 1, "503": 1},
        "successful": 7,
        "success_latency": {"p95_ms": 5},
    }
    result = budgets(stats, {"peak_mib": 1, "growth_budget_passed": None}, 10, 1, 10)
    assert result["other_errors"] == 1
    assert result["checks"]["timeouts"] is False
    assert result["checks"]["cancellations"] is False
    assert result["indeterminate"] is True


def test_custom_query_set_does_not_open_qrels(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps([{"id": "001", "text": "IL-6: 2020?"}]))
    assert query_pool(path)[0]["id"] == "001"
    path.write_text(json.dumps([{"id": "001", "text": " "}]))
    with pytest.raises(ValueError):
        query_pool(path)


def test_object_observation_never_forces_collection(monkeypatch):
    import gc

    def forbidden(*args):
        raise AssertionError("No evidence for forced GC")

    monkeypatch.setattr(gc, "collect", forbidden)
    counts = object_counts()
    assert counts["tracked_objects"] > 0
    assert len(counts["gc_generation_counts"]) == 3


def test_native_class_module_descriptor_does_not_break_object_census(monkeypatch):
    import gc

    class NativeLike:
        pass

    NativeLike.__module__ = property()
    monkeypatch.setattr(gc, "get_objects", lambda: [NativeLike()])
    assert object_counts()["tracked_objects"] == 1


def test_native_available_memory_is_separate_from_process_rss():
    data = system_memory()
    assert 0 < data["system_available_bytes"] <= data["system_total_bytes"]


def test_fixed_duration_stops_submissions_and_drains_without_cancelling(live, tmp_path):
    url, manager, _, _ = live
    query = {"id": "authored", "text": "ERR_42"}
    expected = {
        "rankings": {
            "bm25": {
                "authored": [
                    {"document_id": r.document_id, "rank": r.rank, "score": r.score}
                    for r in manager.get("bm25").search("ERR_42", 10)
                ]
            }
        }
    }
    cycle = [{"method": "bm25", "query": query}]
    captured = []
    rows, stats = asyncio.run(
        duration_batch(
            url, cycle, 0.3, 2, 2, "test", expected, tmp_path / "progress.json", captured.append
        )
    )
    assert rows and len(rows) == len(captured) == stats["requests"]
    assert all(r["start_seconds"] < 0.3 for r in rows)
    assert stats["cancelled"] == 0
    assert all(r["correct"] for r in rows)
    assert manager.snapshot()["activity"]["active"] == 0


def test_punctuation_only_query_uses_existing_frozen_lexical_route(live, tmp_path):
    url, manager, _, _ = live
    expected = {
        "rankings": {"router": {"punct": []}},
        "routing_decisions": {"punct": "bm25"},
        "policy_record_sha256": manager.get("policy")[0]["policy_record_sha256"],
    }
    rows, _ = asyncio.run(
        duration_batch(
            url,
            [{"method": "router", "query": {"id": "punct", "text": "!!!"}}],
            0.2,
            1,
            2,
            "test",
            expected,
            tmp_path / "progress.json",
            lambda row: None,
        )
    )
    assert rows and all(r["correct"] and r["selected_strategy"] == "bm25" for r in rows)


def test_diagnostic_controller_survives_transient_windows_read_denial(tmp_path, monkeypatch):
    import time
    import threading
    import uvicorn
    import app.evaluation.soak_diagnostics as diagnostics

    attempts = []

    def reader(path):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError("temporary atomic-replace sharing denial")
        return {"sequence": 1, "command": "stop"}

    class Server:
        def __init__(self, config):
            self.should_exit = False

        def run(self):
            deadline = time.monotonic() + 2
            while not self.should_exit:
                assert time.monotonic() < deadline
                threading.Event().wait(0.01)

    monkeypatch.setattr(diagnostics, "read_control", reader)
    monkeypatch.setattr(uvicorn, "Server", Server)
    diagnostics.serve(12345, tmp_path / "control.json")
    reply = tmp_path / "command_1.json"
    deadline = time.monotonic() + 2
    while not reply.exists():
        assert time.monotonic() < deadline
        threading.Event().wait(0.01)
    data = json.loads(reply.read_text())
    assert data["status"] == "complete" and data["transient_control_reads"] == 1
