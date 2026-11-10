"""Optional MLflow registration after canonical outputs exist. No inference imports."""

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid
import warnings

from app.tracking.records import (
    SCHEMA,
    ROOT,
    digest,
    load_record,
    metrics,
    parameters,
    sanitize,
    timestamp,
)

_artifact_lock = threading.RLock()


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    uri: str = "http://127.0.0.1:5000"
    experiment: str = "rankforge"
    receipt_directory: Path = ROOT / "experiments/tracking"


def client_for(uri):
    # Optional dependency: never imported when tracking is disabled.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)

    class UriBoundClient:
        def __getattr__(self, name):
            method = getattr(client, name)
            if name not in ("log_artifact", "list_artifacts", "download_artifacts"):
                return method

            def call(*args, **kwargs):
                # Pinned SDK's proxied artifact repository consults the fluent URI.
                # Scope/restore it rather than leaving process-wide configuration changed.
                with _artifact_lock:
                    previous = mlflow.get_tracking_uri()
                    mlflow.set_tracking_uri(uri)
                    try:
                        return method(*args, **kwargs)
                    finally:
                        mlflow.set_tracking_uri(previous)

            return call

    return UriBoundClient()


def register(
    source,
    config=TrackingConfig(),
    *,
    historical=True,
    expected_sha256=None,
    client_factory=client_for,
):
    if not config.enabled:
        return {"status": "disabled"}
    receipt = {
        "status": "failed",
        "historical": historical,
        "source": str(Path(source).resolve()),
        "schema": SCHEMA,
    }
    parent_id = None
    client = None
    children = []
    try:
        record = load_record(source, expected_sha256)
        identity = record.identity()
        receipt.update(source_sha256=record.source_hash, source_bundle_sha256=identity)
        client = client_factory(config.uri)
        experiment_name = config.experiment + "/" + record.family
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id if experiment else client.create_experiment(experiment_name)
        )
        mode = "historical" if historical else "new"
        found = client.search_runs(
            [experiment_id],
            filter_string=f"tags.`rankforge.source_bundle_sha256` = '{identity}' and tags.`rankforge.mode` = '{mode}' and attributes.status = 'FINISHED'",
            max_results=1,
        )
        if found:
            receipt.update(status="duplicate", parent_run_id=found[0].info.run_id)
        else:
            tags = {
                "rankforge.schema": SCHEMA,
                "rankforge.mode": mode,
                "rankforge.family": record.family,
                "rankforge.source_bundle_sha256": identity,
                "rankforge.source_sha256": record.source_hash,
                "rankforge.source_reference": str(record.source),
                "rankforge.original_directory": str(record.artifact_root),
                "rankforge.original_run_id": str(
                    record.metadata.get("run_id", record.artifact_root.name)
                ),
                "rankforge.timing_methodology": str(
                    record.metadata.get(
                        "timing", record.metadata.get("limitations", "Not recorded")
                    )
                )[:4000],
            }
            original_time = timestamp(record.metadata)
            if original_time is not None:
                tags["rankforge.original_timestamp_ms"] = str(original_time)
            parent = client.create_run(
                experiment_id,
                start_time=original_time,
                tags=tags,
                run_name=mode + "-" + tags["rankforge.original_run_id"],
            )
            parent_id = parent.info.run_id
            receipt["parent_run_id"] = parent_id
            with tempfile.TemporaryDirectory(prefix="rankforge-tracking-") as directory:
                staged = Path(directory)
                manifest = record.manifest()
                snapshot = {
                    "source_sha256": record.source_hash,
                    "source_bundle_sha256": identity,
                    "artifact_sha256": manifest,
                    "metadata": sanitize(record.metadata),
                    "missing_values": "Omitted; never filled with zero or inferred from other populations.",
                    "comparison_warning": "Filter by dataset/scope/query set/timing/platform; different groups are descriptive, not equivalent experiments.",
                }
                path = staged / "source_manifest.json"
                path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
                client.log_artifact(parent_id, str(path))
                for artifact in record.artifacts:
                    relative = artifact.relative_to(record.artifact_root)
                    target = staged / "artifacts" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if artifact.suffix == ".json":
                        target.write_text(
                            json.dumps(sanitize(json.loads(artifact.read_text(encoding="utf-8")))),
                            encoding="utf-8",
                        )
                    elif artifact.suffix == ".csv":
                        with artifact.open(encoding="utf-8", newline="") as stream:
                            reader = csv.DictReader(stream)
                            values = [sanitize(row) for row in reader]
                            fields = reader.fieldnames
                        with target.open("w", encoding="utf-8", newline="") as stream:
                            writer = csv.DictWriter(stream, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(values)
                    else:
                        target.write_bytes(artifact.read_bytes())
                    client.log_artifact(
                        parent_id,
                        str(target),
                        artifact_path="artifacts"
                        + ("/" + relative.parent.as_posix() if len(relative.parts) > 1 else ""),
                    )
                # Always retain a sanitized primary report, even outside an artifact directory.
                target = staged / "source_report.json"
                if record.source.suffix == ".json":
                    target.write_text(
                        json.dumps(sanitize(json.loads(record.source.read_text(encoding="utf-8")))),
                        encoding="utf-8",
                    )
                    client.log_artifact(parent_id, str(target))
                for index, row in enumerate(record.rows):
                    method = str(row.get("method", row.get("system", "aggregate")))
                    child = client.create_run(
                        experiment_id,
                        start_time=original_time,
                        tags={
                            "mlflow.parentRunId": parent_id,
                            "rankforge.mode": mode,
                            "rankforge.family": record.family,
                            "rankforge.row_index": str(index),
                            "rankforge.comparison_group": identity,
                            "rankforge.source_sha256": record.source_hash,
                        },
                        run_name=f"{method}-{row.get('condition', record.family)}-{row.get('clients', '')}",
                    )
                    child_id = child.info.run_id
                    children.append(child_id)
                    for key, value in parameters(record, row).items():
                        client.log_param(child_id, key, value)
                    for key, value in metrics(record, row).items():
                        client.log_metric(
                            child_id, key, value, timestamp=original_time or int(time.time() * 1000)
                        )
                    client.set_terminated(child_id, "FINISHED")
                if record.source_hash != digest(record.source) or manifest != record.manifest():
                    raise ValueError("Source changed during tracking upload")
                client.set_tag(parent_id, "rankforge.registration_complete", "true")
                client.set_terminated(parent_id, "FINISHED")
                receipt.update(
                    status="logged", child_run_ids=children, experiment_name=experiment_name
                )
    except Exception as exc:
        receipt["error_type"] = type(exc).__name__
        if client and parent_id:
            try:
                client.set_terminated(parent_id, "FAILED")
            except Exception:
                pass
        warnings.warn(
            f"Optional MLflow registration failed ({type(exc).__name__}); canonical local results remain available.",
            RuntimeWarning,
        )
    try:
        config.receipt_directory.mkdir(parents=True, exist_ok=True)
        path = config.receipt_directory / (uuid.uuid4().hex + ".json")
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        receipt["receipt_path"] = str(path)
    except OSError:
        warnings.warn(
            "Tracking receipt could not be written; local evaluation is unaffected.", RuntimeWarning
        )
    return receipt


def add_arguments(parser):
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Optionally register saved outputs after computation",
    )
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    parser.add_argument("--tracking-experiment", default="rankforge")


def track_arguments(args, source, historical=False):
    result = register(
        source,
        TrackingConfig(args.tracking, args.tracking_uri, args.tracking_experiment),
        historical=historical,
    )
    if args.tracking:
        print(json.dumps({"tracking": result}))
    return result


def main():
    parser = argparse.ArgumentParser(description="Register saved results without model inference")
    parser.add_argument("source", type=Path)
    parser.add_argument("--expect-sha256")
    add_arguments(parser)
    args = parser.parse_args()
    if not args.tracking:
        parser.error("Historical registration requires explicit --tracking")
    result = register(
        args.source,
        TrackingConfig(True, args.tracking_uri, args.tracking_experiment),
        historical=True,
        expected_sha256=args.expect_sha256,
    )
    print(json.dumps(result))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
