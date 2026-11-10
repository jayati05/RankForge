import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.data.loaders import load_dataset
from app.evaluation.comparison import (
    ComparisonConfig,
    analyze_query,
    replay_comparison,
    run_comparison,
)
from app.retrieval.base import SearchResult

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke"


class OfflineEncoder:
    identity = {"model_name": "test-hash-vectors", "revision": "v1"}
    dimension = 4

    def encode(self, texts, batch_size):
        return np.array(
            [[1 + b for b in hashlib.sha256(text.encode()).digest()[:4]] for text in texts]
        )


class OfflineScorer:
    identity = {"model_name": "test-hash-scores", "revision": "v1"}

    def score(self, pairs, batch_size):
        return np.array([hashlib.sha256((q + text).encode()).digest()[0] for q, text in pairs])


def config(tmp_path):
    return ComparisonConfig(
        k=3,
        candidate_depth=5,
        rerank_candidates=3,
        max_queries=2,
        cache=str(tmp_path / "dense.npz"),
    )


def test_all_retrievers_save_and_replay_same_queries(tmp_path):
    run = run_comparison(
        SMOKE, tmp_path / "runs", config(tmp_path), OfflineEncoder(), OfflineScorer()
    )
    assert replay_comparison(run, SMOKE) == 8
    metadata = json.loads((run / "run_metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["baseline_configuration"]["k1"] == 1.5
    assert metadata["dense_cache"]["status"] == "built"
    assert metadata["models"]["dense"] == OfflineEncoder.identity
    rankings = json.loads((run / "rankings.json").read_text())
    assert len({tuple(sorted(values)) for values in rankings.values()}) == 1
    with (run / "per_query_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(int(r["candidate_count"]) == 3 for r in rows if r["method"] == "reranked")
    assert all(int(r["candidate_count"]) == 0 for r in rows if r["method"] != "reranked")
    for required in ["failure_cases.json", "candidate_pools.json", "benchmark_summary.csv"]:
        assert (run / required).is_file()
    again = run_comparison(
        SMOKE, tmp_path / "runs", config(tmp_path), OfflineEncoder(), OfflineScorer()
    )
    assert run != again
    assert json.loads((again / "rankings.json").read_text()) == rankings
    assert json.loads((again / "run_metadata.json").read_text())["dense_cache"]["status"] == "hit"


@pytest.mark.parametrize("tamper", ["metrics", "query_set", "candidate_pool"])
def test_replay_rejects_corruption(tmp_path, tamper):
    run = run_comparison(
        SMOKE, tmp_path / "runs", config(tmp_path), OfflineEncoder(), OfflineScorer()
    )
    if tamper == "metrics":
        path = run / "per_query_results.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["ndcg@3"] = "9"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif tamper == "query_set":
        path = run / "rankings.json"
        values = json.loads(path.read_text())
        values["dense"].pop(next(iter(values["dense"])))
        path.write_text(json.dumps(values))
    else:
        path = run / "candidate_pools.json"
        values = json.loads(path.read_text())
        values[next(iter(values))]["hybrid"] = []
        path.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        replay_comparison(run, SMOKE)


def test_failed_inference_records_incomplete_run(tmp_path):
    class BrokenScorer(OfflineScorer):
        def score(self, pairs, batch_size):
            raise RuntimeError("controlled inference failure")

    with pytest.raises(RuntimeError, match="controlled inference failure"):
        run_comparison(SMOKE, tmp_path / "runs", config(tmp_path), OfflineEncoder(), BrokenScorer())
    run = next((tmp_path / "runs").iterdir())
    metadata = json.loads((run / "run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "controlled inference failure" in metadata["error"]
    assert not (run / "benchmark_summary.csv").exists()
    with pytest.raises(ValueError, match="incomplete"):
        replay_comparison(run, SMOKE)


def test_analysis_improvements_regressions_and_missing_candidates():
    def hits(ids):
        return [SearchResult(did, i, 1.0) for i, did in enumerate(ids, 1)]

    base = hits(["b", "a"])
    rankings = {"bm25": base, "dense": base, "hybrid": base, "reranked": hits(["a", "b"])}
    pools = {
        "hybrid": [{"document_id": "b", "rank": 1}, {"document_id": "a", "rank": 2}],
        "reranked": [{"document_id": "a", "rank": 1}, {"document_id": "b", "rank": 2}],
    }
    improved = analyze_query(rankings, {"a": 1, "missing": 1}, pools, 2)
    assert improved["rerank_effect"] == "improved"
    assert improved["candidate_presence"] == {"a": True, "missing": False}
    assert improved["positions_in_top_k"]["hybrid"]["a"] == 2
    rankings["hybrid"], rankings["reranked"] = rankings["reranked"], rankings["hybrid"]
    assert analyze_query(rankings, {"a": 1}, pools, 2)["rerank_effect"] == "degraded"


def test_percentiles_and_bad_config(tmp_path):
    data = load_dataset(SMOKE)
    assert data.metadata["synthetic"]
    for kwargs in [{"k": 0}, {"rerank_candidates": 1}, {"threads": 0}]:
        settings = config(tmp_path)
        for name, value in kwargs.items():
            setattr(settings, name, value)
        with pytest.raises(ValueError):
            run_comparison(SMOKE, tmp_path / "runs", settings, OfflineEncoder(), OfflineScorer())


def test_latency_percentile_convention():
    from app.evaluation.comparison import METHODS, summaries

    rows = [
        {
            "method": method,
            "latency_ms": i,
            "rerank_ms": 0,
            "candidate_count": 0,
            "recall@10": 1,
            "precision@10": 0.1,
            "mrr@10": 1,
            "ndcg@10": 1,
        }
        for method in METHODS
        for i in range(20)
    ]
    summary = summaries(rows, 10)
    assert summary[0]["p50_latency_ms"] == 9.5
    assert summary[0]["p95_latency_ms"] == pytest.approx(18.05)
    short = [row for row in rows if row["latency_ms"] < 2]
    assert summaries(short, 10)[0]["p95_latency_ms"] is None


def test_report_examples_trace_to_run(tmp_path):
    from app.evaluation.report import export_report

    run = run_comparison(
        SMOKE, tmp_path / "runs", config(tmp_path), OfflineEncoder(), OfflineScorer()
    )
    output = tmp_path / "report.json"
    report = export_report(run, SMOKE, output)
    failures = json.loads((run / "failure_cases.json").read_text())
    for example in report["examples"].values():
        if example:
            assert example["observation"] == failures[example["query_id"]]
    assert report["metadata"]["query_count"] == 2
    with pytest.raises(FileExistsError):
        export_report(run, SMOKE, output)
