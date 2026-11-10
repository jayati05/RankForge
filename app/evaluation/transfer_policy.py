"""Freeze the exact saved Phase 3 policy before external-data preparation."""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.phase3 import read_json
from app.evaluation.routing_experiment import freeze_digest
from app.retrieval.router import RoutingPolicy

POLICY_FILES = ("app/retrieval/router.py", "app/evaluation/features.py")


def load_protocol(root: Path) -> tuple[dict, RoutingPolicy]:
    root = Path(root)
    protocol = read_json(root / "protocol.json")
    raw = (root / "frozen_policy.json").read_bytes()
    record = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest() != protocol["policy_file_sha256"]
        or freeze_digest(record) != protocol["policy_record_sha256"]
    ):
        raise ValueError("Frozen transfer policy was changed")
    policy = RoutingPolicy(**record["policy"])
    if asdict(policy) != record["policy"]:
        raise ValueError("Saved policy cannot be imported unchanged")
    workspace = Path(__file__).resolve().parents[2]
    for name, expected in protocol["policy_implementation_sha256"].items():
        if hashlib.sha256((workspace / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Phase 3 policy implementation changed; transfer must stop")
    return protocol, policy


def freeze_transfer(source: Path, destination: Path) -> Path:
    source, destination = Path(source), Path(destination)
    if destination.exists():
        protocol, _ = load_protocol(destination)
        if protocol["source_phase3"] != str(source.resolve()):
            raise ValueError("Existing protocol is for another source")
        if (source / "frozen_policy.json").read_bytes() != (
            destination / "frozen_policy.json"
        ).read_bytes():
            raise ValueError("Source policy changed since transfer freeze")
        return destination
    metadata = read_json(source / "run_metadata.json")
    if metadata["status"] != "complete":
        raise ValueError("Phase 3 source must be complete")
    raw = (source / "frozen_policy.json").read_bytes()
    record = json.loads(raw)
    if freeze_digest(record) != metadata["frozen_policy_sha256"]:
        raise ValueError("Source Phase 3 policy hash mismatch")
    if asdict(RoutingPolicy(**record["policy"])) != record["policy"]:
        raise ValueError("Source policy cannot be imported unchanged")
    workspace = Path(__file__).resolve().parents[2]
    expected_code = {name: metadata["code_sha256"][name] for name in POLICY_FILES}
    for name, expected in expected_code.items():
        if hashlib.sha256((workspace / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Source policy implementation no longer matches current code")
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_phase3": str(source.resolve()),
        "source_run_id": metadata["run_id"],
        "source_metadata_sha256": hashlib.sha256(
            (source / "run_metadata.json").read_bytes()
        ).hexdigest(),
        "policy_record_sha256": freeze_digest(record),
        "policy_file_sha256": hashlib.sha256(raw).hexdigest(),
        "policy_implementation_sha256": expected_code,
        "pipeline_config": {
            **metadata["pipeline_config"],
            "cache": "indexes/nfcorpus-minilm.npz",
            "require_cache": False,
        },
        "reference_models": metadata.get("live_setup", {}).get("models", {}),
        "external_dataset": "beir-nfcorpus",
        "evaluation_split": "test",
        "rule": "Import exact Phase 3 policy; no external labels for tuning, routing or feature extraction",
        "split_rationale": "Official BEIR test split, selected before label inspection; inspect split availability/counts during validated import, not for tuning",
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "frozen_policy.json").write_bytes(raw)
    (destination / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    load_protocol(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze exact Phase 3 routing policy before NFCorpus import"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/results/phase3/phase3-20261031T065803Z-72804b2af31a"),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/configs/phase4-transfer"))
    args = parser.parse_args()
    try:
        root = freeze_transfer(args.source, args.output)
        print(root)
        print(load_protocol(root)[0]["policy_record_sha256"])
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Transfer freeze failed: {exc}\n")


if __name__ == "__main__":
    main()
