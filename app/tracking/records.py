"""Explicit saved-result adapters. Never import retrievers or load datasets."""

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "rankforge-tracking-v2"
ARTIFACT_NAMES = {
    "run_metadata.json",
    "experiment_metadata.json",
    "benchmark_summary.json",
    "benchmark_summary.csv",
    "per_query_results.csv",
    "system_comparison.csv",
    "per_query_comparison.csv",
    "failure_cases.csv",
    "failure_cases.json",
    "failure_summary.csv",
    "failure_analysis.json",
    "routing_decisions.csv",
    "query_features.csv",
    "routing_evaluation.json",
    "split_metadata.json",
    "frozen_policy.json",
    "transfer_protocol.json",
    "dataset_provenance.json",
    "tuning_audit.json",
    "generalization_analysis.json",
    "phase4_report.json",
    "soak_summary.json",
    "memory_analysis.json",
    "native_memory_summary.json",
    "phase6_report.json",
    "phase6_1_report.json",
    "phase6_2_report.json",
    "phase7a_report.json",
    "docker_build_report.json",
    "container_startup_metrics.json",
    "container_correctness.json",
    "measurements.json",
    "method_latency.csv",
    "latency_by_concurrency.csv",
    "repeated_run_comparison.csv",
    "rss_timeseries.csv",
    "private_commit_timeseries.csv",
    "request_outcomes.csv",
    "container_resource_usage.csv",
    "container_api_latency.csv",
}
SENSITIVE = (
    "password",
    "secret",
    "credential",
    "authorization",
    "api_key",
    "access_token",
    "refresh_token",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sanitize(value, key=""):
    if (
        any(word in key.lower() for word in SENSITIVE)
        or key.lstrip("-").lower() == "token"
        or key.lower().endswith("_token")
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: sanitize(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [
            "[REDACTED]"
            if i
            and isinstance(value[i - 1], str)
            and value[i - 1].startswith("--")
            and sanitize("probe", value[i - 1]) == "[REDACTED]"
            else sanitize(x)
            for i, x in enumerate(value)
        ]
    if isinstance(value, str):
        if (
            "=" in value
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value.split("=", 1)[0])
            and sanitize("probe", value.split("=", 1)[0]) == "[REDACTED]"
        ):
            return value.split("=", 1)[0] + "=[REDACTED]"
        value = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)
        return re.sub(
            r"([?&](?:token|api_key|password)=)[^&\s]+", r"\1[REDACTED]", value, flags=re.I
        )
    return value


@dataclass
class Record:
    source: Path
    artifact_root: Path
    metadata: dict
    rows: list[dict]
    family: str
    source_hash: str
    artifacts: list[Path]

    def manifest(self):
        return {p.relative_to(self.artifact_root).as_posix(): digest(p) for p in self.artifacts}

    def identity(self):
        value = {
            "schema": SCHEMA,
            "source": self.source_hash,
            "artifacts": self.manifest(),
            "normalized_rows": [
                {"parameters": parameters(self, row), "metrics": metrics(self, row)}
                for row in self.rows
            ],
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_artifacts(root):
    result = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            continue
        relative = path.relative_to(root)
        if (
            len(relative.parts) == 1
            and path.name in ARTIFACT_NAMES
            or (relative.parts[0] == "plots" and path.suffix == ".png")
            or (
                len(relative.parts) == 1
                and path.name
                in ("api_latency.png", "memory_usage.png", "startup_initialization.png")
            )
        ):
            if path.stat().st_size <= 20 * 1024**2:
                result.append(path)
    return result


def load_record(source, expected_sha256=None):
    source = Path(source).resolve()
    if source.is_dir():
        root = source
        name = next(
            (
                n
                for n in (
                    "phase4_report.json",
                    "routing_evaluation.json",
                    "benchmark_summary.json",
                    "benchmark_summary.csv",
                )
                if (root / n).is_file()
            ),
            None,
        )
        if name is None:
            raise ValueError("Unsupported saved experiment directory; pass a supported report JSON")
        source = root / name
    else:
        root = source.parent
    checksum = digest(source)
    if expected_sha256 and checksum != expected_sha256:
        raise ValueError("Source SHA256 mismatch")
    if source.suffix == ".csv":
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        data = {"summary": rows}
    else:
        data = read_json(source)
        if isinstance(data, list):
            data = {"summary": data}
    candidate = data.get("artifact_directory")
    if not candidate and str(data.get("run_id", "")).startswith("phase7a-"):
        candidate = ROOT / "experiments/results/phase7a" / data["run_id"]
    if candidate:
        candidate = Path(candidate).resolve()
        if candidate.is_dir() and candidate.is_relative_to(ROOT / "experiments/results"):
            root = candidate
    metadata = data.get("metadata", data.get("run_metadata", data.get("experiment_metadata")))
    if metadata is None:
        metadata_path = root / (
            "experiment_metadata.json"
            if (root / "experiment_metadata.json").is_file()
            else "run_metadata.json"
        )
        metadata = read_json(metadata_path) if metadata_path.is_file() else dict(data)
    if metadata.get("status") not in (None, "complete") or data.get("status") in (
        "failed",
        "running",
        "interrupted",
    ):
        raise ValueError("Only completed saved experiments can be registered")
    metadata = {**metadata}
    metadata.setdefault("dataset", data.get("dataset"))
    if (
        data.get("summary")
        or data.get("routing_evaluation")
        or source.name == "routing_evaluation.json"
    ):
        rows = data.get("summary", data.get("routing_evaluation", {}).get("summary", []))
        if not rows:
            raise ValueError("No saved evaluation summary")
        family = "evaluation"
        if metadata.get("source_phase2"):
            saved = Path(metadata["source_phase2"]) / "run_metadata.json"
            if saved.is_file() and saved.resolve().is_relative_to(ROOT / "experiments/results"):
                previous = read_json(saved)
                for key in ("dataset", "document_count", "baseline_configuration", "device"):
                    if metadata.get(key) is None:
                        metadata[key] = previous.get(key)
            metadata["evaluation_scope"] = "internal_held_out_from_previously_exposed_test"
        else:
            metadata["evaluation_scope"] = "original_saved_split"
    else:
        family = "operational"
        rows = []
        stages = data.get("summaries", data.get("load_summary", []))
        if data.get("runs"):
            stages = [
                {**r["statistics"], "stage": r["run"], "config": r.get("config", {})}
                for r in data["runs"]
                if "statistics" in r
            ]
        for stage in stages:
            if "by_method" in stage:
                for method, values in stage["by_method"].items():
                    rows.append(
                        {
                            **values,
                            "method": method,
                            "condition": stage.get("stage"),
                            "clients": stage.get("concurrency"),
                            "config": stage.get("config", {}),
                        }
                    )
            else:
                operational_row = {
                    **stage,
                    "condition": stage.get("label", stage.get("stage")),
                    "clients": stage.get("clients", stage.get("concurrency")),
                }
                label = str(operational_row.get("condition", ""))
                for case in data.get("containers", []):
                    if label in (case["label"], case["label"] + "_restart"):
                        operational_row.update(
                            memory_limit=case["memory_limit"],
                            cpu_limit=case["cpu_limit"],
                            experimental_blas_threads=case.get("experimental_blas_threads"),
                            execution_environment="container",
                            platform=case.get("before_stop", {}).get("Platform", "linux"),
                        )
                        image_env = case.get("before_stop", {}).get("Config", {}).get("Env", [])
                        operational_row["python"] = next(
                            (
                                v.split("=", 1)[1]
                                for v in image_env
                                if v.startswith("PYTHON_VERSION=")
                            ),
                            None,
                        )
                        operational_row["image_id"] = case.get("before_stop", {}).get("Image")
                        freeze = data.get("lifecycle_probe", {}).get("image_dependency_freeze", "")
                        operational_row["versions"] = dict(
                            line.split("==", 1)
                            for line in freeze.splitlines()
                            if "==" in line and not line.startswith("#")
                        )
                if label == "local":
                    operational_row["execution_environment"] = "native"
                rows.append(operational_row)
        if not rows:
            raise ValueError("Unsupported report: no measured operational summaries")
    # Explicit data/setup snapshots, never model files or raw corpus.
    metadata["tracking_setup"] = data.get(
        "setup", metadata.get("live_setup", metadata.get("setup", {}))
    )
    artifacts = select_artifacts(root)
    return Record(source, root, metadata, rows, family, checksum, artifacts)


def numeric(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def timestamp(metadata):
    for key in ("timestamp_utc", "created_at_utc", "started_utc", "report_published_at_utc"):
        if metadata.get(key):
            return int(
                datetime.fromisoformat(metadata[key].replace("Z", "+00:00")).timestamp() * 1000
            )
    return None


def parameters(record, row):
    m = record.metadata
    config = m.get("pipeline_config", m.get("evaluation_configuration", m.get("config", {})))
    dataset = m.get("dataset") or {}
    if not isinstance(dataset, dict):
        dataset = {"name": dataset}
    result = {
        "dataset.name": dataset.get("name"),
        "dataset.version": dataset.get("version"),
        "dataset.source": dataset.get("source_url"),
        "dataset.split": config.get("split", m.get("split", m.get("evaluation_split"))),
        "corpus_size": m.get("document_count", dataset.get("documents")),
        "query_count": row.get("query_count", m.get("query_count")),
        "judgment_count": m.get("judgment_count"),
        "method": row.get("method", row.get("system")),
        "top_k": config.get("k", m.get("k")),
        "seed": config.get("seed", m.get("seed", m.get("subset", {}).get("seed"))),
        "device": m.get("device"),
        "platform": row.get("platform", m.get("platform")),
        "python": row.get("python", m.get("python")),
        "report_host_python": m.get("python")
        if row.get("execution_environment") == "container"
        else None,
        "image_id": row.get("image_id"),
        "condition": row.get("condition"),
        "clients": row.get("clients"),
        "evaluation_scope": m.get("evaluation_scope"),
        "execution_environment": row.get("execution_environment"),
        "memory_limit": row.get("memory_limit"),
        "cpu_limit": row.get("cpu_limit"),
        "experimental_blas_threads": row.get("experimental_blas_threads"),
    }
    for name in ("candidate_depth", "rrf_constant", "rerank_candidates", "batch_size", "threads"):
        result[name] = config.get(name)
    for name, value in m.get(
        "baseline_configuration", m.get("config", {}) if m.get("system") == "BM25Okapi" else {}
    ).items():
        if name in ("k1", "b", "epsilon", "tokenizer"):
            result["bm25." + name] = value
    models = m.get("models", m.get("tracking_setup", {}).get("models", {}))
    for role, model in models.items():
        result[f"model.{role}.identifier"] = model.get("model_name")
        result[f"model.{role}.revision"] = model.get("revision")
    for name, value in row.get("versions", m.get("versions", {})).items():
        result["software." + name] = value
    return {
        key: str(sanitize(value, key))[:6000] for key, value in result.items() if value is not None
    }


def metrics(record, row):
    result = {}
    if record.family == "evaluation":
        for key, value in row.items():
            if any(key.startswith(name + "@") for name in ("recall", "precision", "mrr", "ndcg")):
                result["eval." + key.replace("@", "_at_")] = value
        for key in (
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "latency_ms",
        ):
            if key in row:
                result["eval." + ("mean_latency_ms" if key == "latency_ms" else key)] = row[key]
    else:
        for key in (
            "requests",
            "successful",
            "failed",
            "timed_out",
            "cancelled",
            "elapsed_seconds",
            "successful_per_second",
            "attempts_per_second",
        ):
            if key in row:
                result["ops." + key] = row[key]
        for category in ("success_latency", "all_latency"):
            for key, value in row.get(category, {}).items():
                result[f"ops.{category}.{key}"] = value
        if "429" in row.get("statuses", {}):
            result["ops.http_429_count"] = row["statuses"]["429"]
    setup = record.metadata.get("tracking_setup", {})
    for key, value in setup.items():
        if key.endswith("_seconds"):
            result["setup." + key] = value
    m = record.metadata
    if "index_build_seconds" in m:
        result["setup.index_build_seconds"] = m["index_build_seconds"]
    for category in ("model_load_seconds", "index_setup_seconds"):
        for key, value in m.get(category, {}).items():
            result[f"setup.{category}.{key}"] = value
    cached = m.get("dense_cache", {})
    if "original_build_seconds" in cached:
        result["setup.original_embedding_build_seconds"] = cached["original_build_seconds"]
    return {key: numeric(value) for key, value in result.items() if numeric(value) is not None}
