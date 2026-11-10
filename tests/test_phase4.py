"""Offline external-validation tests with authored BEIR-format data, not NFCorpus samples."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.data import nfcorpus
from app.data.loaders import load_dataset
from app.evaluation.comparison import METHODS, ComparisonConfig, run_comparison
from app.evaluation.failure_lab import category_summary
from app.evaluation.metrics import ranking_metrics
from app.evaluation.phase3 import read_json, run_phase3
from app.evaluation.phase4 import REQUIRED, replay_phase4, run_phase4
from app.evaluation.routing_experiment import freeze_digest
from app.evaluation.transfer_policy import freeze_transfer, load_protocol
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import RerankingRetriever

SMOKE = Path(__file__).resolve().parents[1] / "data/smoke"


class Encoder:
    dimension = 4
    identity = {"model_name": "authored-phase4-vectors", "revision": "unit"}

    def encode(self, texts, batch_size):
        return np.array([[1 + b for b in hashlib.sha256(t.encode()).digest()[:4]] for t in texts])


class Scorer:
    identity = {"model_name": "authored-phase4-scores", "revision": "unit"}

    def score(self, pairs, batch_size):
        return np.array([hashlib.sha256((q + t).encode()).digest()[0] for q, t in pairs])


@pytest.fixture
def reference(tmp_path):
    root = tmp_path / "reference_data"
    shutil.copytree(SMOKE, root)
    with (root / "queries.jsonl").open("a", encoding="utf-8") as f:
        for q in ("q5", "q6"):
            f.write(json.dumps({"_id": q, "text": "Python virtual environments"}) + "\n")
    with (root / "qrels/test.tsv").open("a") as f:
        f.write("q5\td1\t1\nq6\td1\t1\n")
    data = load_dataset(root)
    encoder, scorer = Encoder(), Scorer()
    source2 = run_comparison(
        root,
        tmp_path / "source2",
        ComparisonConfig(k=3, candidate_depth=5, rerank_candidates=3, cache=None),
        encoder,
        scorer,
    )
    bm25 = BM25Retriever(data.documents.values())
    dense = DenseRetriever(data.documents.values(), encoder)
    hybrid = HybridRetriever(bm25, dense, 5)
    reranked = RerankingRetriever(hybrid, data.documents.values(), scorer, 3)
    systems = dict(zip(METHODS, (bm25, dense, hybrid, reranked)))
    source3 = run_phase3(root, source2, tmp_path / "source3", systems_override=systems, plots=False)
    protocol = freeze_transfer(source3, tmp_path / "protocol")
    report2 = tmp_path / "report2.json"
    report3 = tmp_path / "report3.json"
    report2.write_text(
        json.dumps(
            {
                "summary": read_json(source2 / "benchmark_summary.json"),
                "metadata": read_json(source2 / "run_metadata.json"),
            }
        )
    )
    report3.write_text(
        json.dumps(
            {
                "routing_evaluation": read_json(source3 / "routing_evaluation.json"),
                "failure_summary": category_summary(read_json(source3 / "failure_analysis.json")),
            }
        )
    )
    return {
        "root": root,
        "protocol": protocol,
        "source3": source3,
        "systems": systems,
        "report2": report2,
        "report3": report3,
    }


@pytest.fixture
def archive(reference, tmp_path, monkeypatch):
    archive = tmp_path / "authored-nfcorpus.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("corpus.jsonl", "queries.jsonl"):
            z.writestr("nfcorpus/" + name, (reference["root"] / name).read_bytes())
        z.writestr("nfcorpus/qrels/train.tsv", "query-id\tcorpus-id\tscore\nq1\td2\t1\n")
        z.writestr("nfcorpus/qrels/dev.tsv", "query-id\tcorpus-id\tscore\nq2\td3\t2\n")
        z.writestr(
            "nfcorpus/qrels/test.tsv",
            "query-id\tcorpus-id\tscore\nq3\td5\t2\nq3\td4\t1\nq4\td1\t1\nq5\td1\t1\nq6\td1\t1\n",
        )
        z.writestr("../../outside.txt", "never extract")
    monkeypatch.setattr(nfcorpus, "NFCORPUS_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    return archive


def test_frozen_import_exact_and_immutable(reference):
    root = reference["protocol"]
    assert (root / "frozen_policy.json").read_bytes() == (
        reference["source3"] / "frozen_policy.json"
    ).read_bytes()
    protocol, policy = load_protocol(root)
    assert freeze_digest(read_json(root / "frozen_policy.json")) == protocol["policy_record_sha256"]
    assert freeze_transfer(reference["source3"], root) == root
    record = read_json(root / "frozen_policy.json")
    record["policy"]["rerank_max_tokens"] += 1
    (root / "frozen_policy.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="changed"):
        load_protocol(root)


def test_nfcorpus_import_preserves_grades_splits_raw_and_ids(reference, archive, tmp_path):
    target = tmp_path / "external"
    before = archive.read_bytes()
    nfcorpus.prepare_nfcorpus(archive, target, reference["protocol"])
    assert archive.read_bytes() == before
    assert len(load_dataset(target, "test").queries) == 4
    assert load_dataset(target, "test").qrels["q3"] == {"d5": 2, "d4": 1}
    assert list(load_dataset(target, "dev").queries) == ["q2"]
    meta = read_json(target / "metadata.json")
    assert meta["raw_query_count"] == 6 and meta["document_count"] == 6
    assert meta["raw_query_ids_without_any_qrel"] == []
    assert meta["source_archive_sha256"] == hashlib.sha256(before).hexdigest()
    with zipfile.ZipFile(archive) as z:
        for name in nfcorpus.MEMBERS:
            assert (target / name).read_bytes() == z.read("nfcorpus/" + name)
    assert not (tmp_path / "outside.txt").exists()
    assert nfcorpus.prepare_nfcorpus(archive, target, reference["protocol"]) == target


def test_checksum_error_precedes_import(reference, archive, tmp_path):
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        nfcorpus.prepare_nfcorpus(archive, tmp_path / "target", reference["protocol"])
    assert not (tmp_path / "target").exists()


def test_policy_required_before_external_label_read(archive, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("labels must not load before freeze")

    monkeypatch.setattr(nfcorpus, "load_dataset", forbidden)
    with pytest.raises(FileNotFoundError):
        nfcorpus.prepare_nfcorpus(archive, tmp_path / "target", tmp_path / "missing_protocol")


def test_join_schema_rejection(reference, archive, tmp_path, monkeypatch):
    altered = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(altered, "w") as dest:
        for name in source.namelist():
            content = source.read(name)
            if name == "nfcorpus/qrels/test.tsv":
                content = content.replace(b"d5", b"unknown-doc")
            dest.writestr(name, content)
    monkeypatch.setattr(nfcorpus, "NFCORPUS_MD5", hashlib.md5(altered.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="unknown query or document"):
        nfcorpus.prepare_nfcorpus(altered, tmp_path / "target", reference["protocol"])
    assert not (tmp_path / "target").exists()


def test_graded_metric_compatibility():
    values = ranking_metrics(["low", "high"], {"high": 2, "low": 1}, 2)
    assert values["recall@2"] == 1 and values["precision@2"] == 1 and values["mrr@2"] == 1
    assert values["ndcg@2"] == pytest.approx((1 + 3 / np.log2(3)) / (3 + 1 / np.log2(3)))


def test_external_evaluation_replays_without_tuning(reference, archive, tmp_path, monkeypatch):
    target = nfcorpus.prepare_nfcorpus(archive, tmp_path / "external", reference["protocol"])
    old_policy = (reference["protocol"] / "frozen_policy.json").read_bytes()
    from app.evaluation import routing_experiment

    def forbidden(*args, **kwargs):
        raise AssertionError("External policy tuning prohibited")

    monkeypatch.setattr(routing_experiment, "select_policy", forbidden)
    run = run_phase4(
        target,
        reference["protocol"],
        tmp_path / "results",
        reference["report2"],
        reference["report3"],
        reference["systems"],
        plots=False,
    )
    assert all((run / name).is_file() for name in REQUIRED)
    assert replay_phase4(run, target) == 20
    assert (reference["protocol"] / "frozen_policy.json").read_bytes() == old_policy
    report = read_json(run / "phase4_report.json")
    assert report["tuning_performed"] is False
    assert report["dataset"]["splits"]["test"]["judgment_count"] == 5
    assert read_json(run / "failure_analysis.json")["q3"]["judgments"]["d5"] == 2
    with (run / "system_comparison.csv").open("a") as f:
        f.write("corrupt")
    with pytest.raises(ValueError, match="artifact hash changed"):
        replay_phase4(run, target)


def test_policy_file_mutation_during_evaluation_is_rejected(
    reference, archive, tmp_path, monkeypatch
):
    from app.evaluation import phase4

    target = nfcorpus.prepare_nfcorpus(archive, tmp_path / "external", reference["protocol"])
    original = phase4.evaluate_live

    def mutate_after_inference(*args, **kwargs):
        result = original(*args, **kwargs)
        record = read_json(reference["protocol"] / "frozen_policy.json")
        record["policy"]["bm25_coverage_min"] = 0.5
        (reference["protocol"] / "frozen_policy.json").write_text(json.dumps(record))
        return result

    monkeypatch.setattr(phase4, "evaluate_live", mutate_after_inference)
    with pytest.raises(ValueError, match="Frozen transfer policy was changed"):
        run_phase4(
            target,
            reference["protocol"],
            tmp_path / "failed_results",
            reference["report2"],
            reference["report3"],
            reference["systems"],
            plots=False,
        )
    root = next((tmp_path / "failed_results").iterdir())
    assert read_json(root / "experiment_metadata.json")["status"] == "failed"
    assert not (root / "phase4_report.json").exists()
