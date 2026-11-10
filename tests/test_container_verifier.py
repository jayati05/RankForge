"""Result validation and empty-artifact contracts without a Docker dependency."""

import csv
import importlib.util
from pathlib import Path


def verifier():
    source = Path(__file__).resolve().parents[1] / "scripts/verify_container.py"
    spec = importlib.util.spec_from_file_location("container_verifier", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equivalence_preserves_strict_score_and_frozen_policy():
    check = verifier().check_result
    expected = [{"document_id": "original-id", "rank": 1, "score": 0.25}]
    actual = {
        "method": "router",
        "documents": expected,
        "routing": {"policy_record_sha256": "frozen"},
    }
    result = check(actual, expected, "frozen")
    assert result["ordered_ids_ranks_equal"] and result["scores_within_1e_5"]
    assert result["frozen_policy_equal"] and result["max_abs_score_difference"] == 0
    actual["documents"] = [{**expected[0], "score": 0.25002}]
    result = check(actual, expected, "changed")
    assert result["ordered_ids_ranks_equal"]
    assert not result["scores_within_1e_5"] and not result["frozen_policy_equal"]
    actual["documents"] = [{**expected[0], "document_id": "different-id"}]
    assert not check(actual, expected, "frozen")["ordered_ids_ranks_equal"]


def test_failed_run_empty_csv_has_schema(tmp_path):
    path = tmp_path / "empty.csv"
    verifier().csv_file(path, [], ["status", "latency_ms"])
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["status", "latency_ms"]
        assert list(reader) == []


def test_resource_analysis_distinguishes_cgroup_and_process_memory():
    source = Path(__file__).resolve().parents[1] / "scripts/summarize_container.py"
    spec = importlib.util.spec_from_file_location("container_summary", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {
            "label": "container",
            "elapsed_seconds": "0",
            "process_rss_bytes": "100",
            "memory.current": "400",
            "memory.peak": "500",
            "memory.max": "1000",
            "cpu.max": "200000 100000",
            "cpu.stat": '{"usage_usec": "1000000"}',
            "memory.events": '{"oom_kill":"0"}',
        },
        {
            "label": "container",
            "elapsed_seconds": "2",
            "process_rss_bytes": "200",
            "memory.current": "450",
            "memory.peak": "600",
            "memory.max": "1000",
            "cpu.max": "200000 100000",
            "cpu.stat": "{'usage_usec': '5000000'}",
            "memory.events": '{"oom_kill":"0"}',
        },
    ]
    actual = module.resource_summary(rows)["container"]
    assert actual["sampled_process_peak_rss_bytes"] == 200
    assert actual["sampled_cgroup_peak_current_bytes"] == 450
    assert actual["cgroup_lifetime_peak_bytes"] == 600
    assert actual["memory_limit_bytes"] == 1000
    assert actual["mean_cpu_percent_one_core_100"] == 200
    assert actual["sampled_private_commit_peak_bytes"] is None
