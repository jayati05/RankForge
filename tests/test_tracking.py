"""Tracking tests use an in-memory client double, without MLflow or Docker."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.tracking.mlflow import TrackingConfig, add_arguments, register
from app.tracking.records import load_record, metrics, parameters, sanitize, select_artifacts


class Client:
    def __init__(self):
        self.runs = {}
        self.uploads = {}

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1")

    def search_runs(self, ids, filter_string, max_results):
        import re

        identity, mode = re.findall(r"= '([^']+)'", filter_string)[:2]
        return [
            r
            for r in self.runs.values()
            if r.info.status == "FINISHED"
            and r.data.tags.get("rankforge.source_bundle_sha256") == identity
            and r.data.tags.get("rankforge.mode") == mode
        ][:max_results]

    def create_run(self, experiment_id, start_time=None, tags=None, run_name=None):
        run = SimpleNamespace(
            info=SimpleNamespace(
                run_id=str(len(self.runs)), status="RUNNING", start_time=start_time
            ),
            data=SimpleNamespace(tags=tags or {}, params={}, metrics={}),
        )
        self.runs[run.info.run_id] = run
        return run

    def log_param(self, run_id, key, value):
        self.runs[run_id].data.params[key] = value

    def log_metric(self, run_id, key, value, timestamp=None):
        self.runs[run_id].data.metrics[key] = value

    def log_artifact(self, run_id, path, artifact_path=None):
        self.uploads[(run_id, (artifact_path + "/" if artifact_path else "") + Path(path).name)] = (
            Path(path).read_bytes()
        )

    def set_tag(self, run_id, key, value):
        self.runs[run_id].data.tags[key] = value

    def set_terminated(self, run_id, status):
        self.runs[run_id].info.status = status


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    metadata = {
        "run_id": "authored-v1",
        "timestamp_utc": "2025-01-01T12:00:00+00:00",
        "status": "complete",
        "dataset": {"name": "authored", "version": "v1"},
        "document_count": 2,
        "query_count": 1,
        "judgment_count": 1,
        "config": {"k": 10, "split": "test", "seed": 42},
        "baseline_configuration": {"k1": 1.5},
        "api_key": "DO-NOT-UPLOAD",
    }
    (root / "run_metadata.json").write_text(json.dumps(metadata))
    (root / "benchmark_summary.json").write_text(
        json.dumps(
            [
                {
                    "method": "bm25",
                    "query_count": 1,
                    "recall@10": 1.0,
                    "precision@10": 0.1,
                    "mrr@10": 0.5,
                    "ndcg@10": 0.6309297535714575,
                    "mean_latency_ms": 2.5,
                    "p99_latency_ms": None,
                }
            ]
        )
    )
    (root / "per_query_results.csv").write_text("query_id,ndcg@10\nq1,0.6309297535714575\n")
    (root / "corpus.jsonl").write_text("RAW-DATA-NO")
    (root / "model.safetensors").write_bytes(b"MODEL-NO")
    (root / ".env").write_text("SECRET=NO")
    return root


def config(tmp_path):
    return TrackingConfig(True, receipt_directory=tmp_path / "receipts")


def test_disabled_never_imports_client_or_reads_source(tmp_path):
    def forbidden(uri):
        raise AssertionError("Client must not be instantiated")

    assert register(tmp_path / "missing", client_factory=forbidden) == {"status": "disabled"}
    assert not (tmp_path / "receipts").exists()


def test_parameters_and_metrics_are_source_values(saved):
    record = load_record(saved)
    values = parameters(record, record.rows[0])
    assert values["dataset.name"] == "authored" and values["top_k"] == "10"
    assert values["bm25.k1"] == "1.5" and values["judgment_count"] == "1"
    values = metrics(record, record.rows[0])
    assert values["eval.ndcg_at_10"] == 0.6309297535714575
    assert "eval.p99_latency_ms" not in values
    assert all(not k.startswith("ops.") for k in values)


def test_register_artifacts_historical_time_and_duplicate(saved, tmp_path):
    client = Client()
    before = {p.name: p.read_bytes() for p in saved.iterdir()}
    first = register(saved, config(tmp_path), client_factory=lambda _: client)
    assert first["status"] == "logged"
    parent = client.runs[first["parent_run_id"]]
    assert parent.info.start_time == 1735732800000
    assert parent.data.tags["rankforge.mode"] == "historical"
    child = client.runs[first["child_run_ids"][0]]
    assert (
        child.data.metrics["eval.ndcg_at_10"]
        == json.loads((saved / "benchmark_summary.json").read_text())[0]["ndcg@10"]
    )
    uploaded = b"".join(client.uploads.values())
    assert b"DO-NOT-UPLOAD" not in uploaded
    assert b"RAW-DATA-NO" not in uploaded and b"MODEL-NO" not in uploaded
    assert any(key[1].endswith("per_query_results.csv") for key in client.uploads)
    second = register(saved, config(tmp_path), client_factory=lambda _: client)
    assert second["status"] == "duplicate" and second["parent_run_id"] == first["parent_run_id"]
    assert len(client.runs) == 2
    assert before == {p.name: p.read_bytes() for p in saved.iterdir()}
    fresh = register(saved, config(tmp_path), historical=False, client_factory=lambda _: client)
    assert (
        fresh["status"] == "logged"
        and client.runs[fresh["parent_run_id"]].data.tags["rankforge.mode"] == "new"
    )


def test_unavailable_does_not_fail_evaluation(saved, tmp_path):
    def unavailable(uri):
        raise ConnectionError("secret endpoint")

    with pytest.warns(RuntimeWarning, match="canonical local results"):
        result = register(saved, config(tmp_path), client_factory=unavailable)
    assert result["status"] == "failed"
    assert result["error_type"] == "ConnectionError"
    assert "secret endpoint" not in json.dumps(result)
    assert list((tmp_path / "receipts").glob("*.json"))


def test_source_hash_and_mutation_checked(saved, tmp_path):
    client = Client()
    with pytest.warns(RuntimeWarning):
        result = register(
            saved, config(tmp_path), expected_sha256="0" * 64, client_factory=lambda _: client
        )
    assert result["status"] == "failed" and not client.runs
    old = load_record(saved).identity()
    (saved / "per_query_results.csv").write_text("changed")
    assert load_record(saved).identity() != old
    checksum = hashlib.sha256((saved / "benchmark_summary.json").read_bytes()).hexdigest()
    assert load_record(saved, checksum).source_hash == checksum


def test_artifact_filter_and_secret_sanitization(saved):
    (saved / "cache").mkdir()
    (saved / "cache" / "run_metadata.json").write_text("CACHE-NO")
    assert {p.name for p in select_artifacts(saved)} == {
        "run_metadata.json",
        "benchmark_summary.json",
        "per_query_results.csv",
    }
    sanitized = sanitize(
        {
            "tokenizer": "unchanged",
            "password": "hidden",
            "command": ["--token", "hidden"],
            "uri": "http://name:password@host/path?token=hidden",
            "env": ["AWS_SECRET_ACCESS_KEY=hidden"],
        }
    )
    assert sanitized["tokenizer"] == "unchanged"
    assert "hidden" not in json.dumps(sanitized) and "password@host" not in json.dumps(sanitized)


def test_operational_metrics_never_become_quality_metrics(tmp_path):
    path = tmp_path / "service.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "summaries": [
                    {
                        "method": "reranked",
                        "label": "container",
                        "clients": 2,
                        "requests": 10,
                        "successful": 7,
                        "statuses": {"200": 7, "429": 3},
                        "success_latency": {"mean_ms": 100, "p95_ms": 150},
                    }
                ],
            }
        )
    )
    record = load_record(path)
    actual = metrics(record, record.rows[0])
    assert actual["ops.http_429_count"] == 3
    assert actual["ops.success_latency.p95_ms"] == 150
    assert not any(k.startswith("eval.") for k in actual)


def test_optional_cli_and_compose_profile():
    import yaml

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert not parser.parse_args([]).tracking
    assert parser.parse_args(["--tracking"]).tracking
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "compose.yaml").read_text())
    server = raw["services"]["mlflow"]
    assert server["profiles"] == ["tracking"]
    assert server["ports"][0].startswith("127.0.0.1:")
    assert len(server["volumes"]) == 2 and "healthcheck" in server
    assert server["restart"] == "unless-stopped"
    assert "depends_on" not in raw["services"]["api"]
    assert "MLFLOW_TRACKING_URI" not in raw["services"]["api"]["environment"]


def test_uri_scoped_artifact_sdk_workaround_restores_global_uri(monkeypatch):
    import sys
    from types import ModuleType
    from app.tracking.mlflow import client_for

    state = {"uri": "file:///original"}
    package = ModuleType("mlflow")
    package.__path__ = []
    package.get_tracking_uri = lambda: state["uri"]
    package.set_tracking_uri = lambda value: state.update(uri=value)
    tracking = ModuleType("mlflow.tracking")

    class SDK:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://127.0.0.1:5000"

        def log_artifact(self, *args, **kwargs):
            assert state["uri"] == "http://127.0.0.1:5000"
            raise RuntimeError("authored upload failure")

    tracking.MlflowClient = SDK
    monkeypatch.setitem(sys.modules, "mlflow", package)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)
    monkeypatch.delenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", raising=False)
    client = client_for("http://127.0.0.1:5000")
    with pytest.raises(RuntimeError):
        client.log_artifact("run", "file")
    assert state["uri"] == "file:///original"
    import os

    assert os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] == "true"


def test_offline_benchmark_works_with_mlflow_import_forbidden(monkeypatch, tmp_path):
    import builtins
    from app.evaluation.benchmark import run_benchmark

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise AssertionError("Offline computation must not import MLflow")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    source = run_benchmark(Path(__file__).resolve().parents[1] / "data/smoke", tmp_path / "results")
    assert {p.name for p in source.iterdir()} == {
        "benchmark_summary.csv",
        "per_query_results.csv",
        "rankings.json",
        "run_metadata.json",
    }
    assert register(source) == {"status": "disabled"}


def test_failed_upload_stays_failed_and_retry_is_not_deduplicated(saved, tmp_path):
    client = Client()
    original = client.log_artifact

    def fail(*args, **kwargs):
        raise OSError("authored network interruption")

    client.log_artifact = fail
    with pytest.warns(RuntimeWarning):
        failed = register(saved, config(tmp_path), client_factory=lambda _: client)
    assert client.runs[failed["parent_run_id"]].info.status == "FAILED"
    client.log_artifact = original
    retried = register(saved, config(tmp_path), client_factory=lambda _: client)
    assert retried["status"] == "logged"
    assert retried["parent_run_id"] != failed["parent_run_id"]


def test_container_parameters_do_not_mislabel_host_python(tmp_path):
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "status": "complete",
                "python": "Windows Python 3.11.5",
                "summaries": [
                    {"method": "bm25", "label": "container_0", "clients": 1, "requests": 1}
                ],
                "containers": [
                    {
                        "label": "container_0",
                        "memory_limit": "4g",
                        "cpu_limit": "4",
                        "before_stop": {
                            "Platform": "linux",
                            "Image": "sha256:authored",
                            "Config": {"Env": ["PYTHON_VERSION=3.11.14"]},
                        },
                    }
                ],
            }
        )
    )
    record = load_record(source)
    actual = parameters(record, record.rows[0])
    assert actual["python"] == "3.11.14"
    assert actual["report_host_python"] == "Windows Python 3.11.5"
    assert actual["image_id"] == "sha256:authored"
    assert actual["platform"] == "linux" and actual["cpu_limit"] == "4"


def test_missing_optional_sdk_is_nonfatal(saved, tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ModuleNotFoundError("authored optional SDK absence")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.warns(RuntimeWarning):
        receipt = register(saved, config(tmp_path))
    assert receipt["status"] == "failed" and receipt["error_type"] == "ModuleNotFoundError"
    assert (saved / "benchmark_summary.json").is_file()


def test_source_mutation_during_upload_fails_registration(saved, tmp_path):
    client = Client()
    original = client.log_artifact
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        original(*args, **kwargs)
        if not changed:
            (saved / "per_query_results.csv").write_text("changed during upload")
            changed = True

    client.log_artifact = mutate
    with pytest.warns(RuntimeWarning):
        receipt = register(saved, config(tmp_path), client_factory=lambda _: client)
    assert receipt["status"] == "failed" and receipt["error_type"] == "ValueError"
    assert client.runs[receipt["parent_run_id"]].info.status == "FAILED"
