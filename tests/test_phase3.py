"""Small authored fixtures test Phase 3 without public data or neural downloads."""

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from app.data.loaders import load_dataset
from app.data.schemas import Document
from app.evaluation.comparison import ComparisonConfig, METHODS, run_comparison
from app.evaluation.failure_lab import classify_failures, query_comparison
from app.evaluation.features import CorpusFeatures, QueryFeatures, ranking_diagnostics
from app.evaluation.phase3 import REQUIRED, read_json, replay_phase3, run_phase3
from app.evaluation.routing_experiment import paired_bootstrap, select_policy
from app.evaluation.splits import exploratory_split, validate_partition
from app.retrieval.base import SearchResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import RerankingRetriever
from app.retrieval.router import QueryAwareRouter, RoutingPolicy

SMOKE = Path(__file__).resolve().parents[1] / "data/smoke"


def hits(ids):
    return [SearchResult(did, i, 1.0 / i) for i, did in enumerate(ids, 1)]


class Encoder:
    identity = {"model_name": "authored-test-vectors", "revision": "v1"}
    dimension = 4

    def encode(self, texts, batch_size):
        return np.array([[1 + b for b in hashlib.sha256(t.encode()).digest()[:4]] for t in texts])


class Scorer:
    identity = {"model_name": "authored-test-scores", "revision": "v1"}

    def score(self, pairs, batch_size):
        return np.array([hashlib.sha256((q + t).encode()).digest()[0] for q, t in pairs])


def test_query_features_manual_values():
    corpus = CorpusFeatures([Document("a", "", "cat cat"), Document("b", "", "dog")])
    f = corpus.extract("cat cat 42!")
    assert asdict(f) == {
        "token_count": 3,
        "unique_token_count": 2,
        "average_token_length": 8 / 3,
        "numeric_indicator": True,
        "punctuation_indicator": True,
        "rare_term_proportion": 0.5,
        "corpus_term_coverage": 0.5,
    }
    assert corpus.extract(" ").token_count == 0
    assert corpus.extract("!!!").punctuation_indicator
    assert corpus.extract("unknown").corpus_term_coverage == 0
    with pytest.raises(ValueError):
        corpus.extract(None)
    with pytest.raises(ValueError):
        CorpusFeatures([])


def test_score_diagnostics_are_label_free():
    f = ranking_diagnostics(hits(["a", "b"]), hits(["b", "c"]))
    assert f["candidate_pool_overlap_topk"] == 0.5
    assert f["topk_jaccard"] == pytest.approx(1 / 3)
    assert f["ranking_position_agreement"] == 0
    assert f["bm25_score_mean"] == 0.75
    assert ranking_diagnostics([], [])["dense_score_mean"] is None


def test_taxonomy_evidence_and_multiple_indicators():
    corpus = CorpusFeatures(
        [Document("a", "", "alpha"), Document("b", "", "beta"), Document("c", "", "gamma")]
    )
    rankings = {
        "bm25": hits(["a"]),
        "dense": hits(["b"]),
        "hybrid": hits(["a"]),
        "reranked": hits(["b"]),
    }
    indicators = classify_failures("beta", rankings, {"a": 1, "c": 1}, hits(["a", "b"]), corpus, 1)
    by_name = {i["category"]: i for i in indicators}
    assert {
        "candidate_generation_failure",
        "reranker_regression",
        "semantic_mismatch",
        "lexical_mismatch",
    } <= set(by_name)
    assert by_name["candidate_generation_failure"]["evidence"]["missing_judged_positive_ids"] == [
        "c"
    ]
    assert by_name["reranker_regression"]["evidence_type"] == "observed"
    assert by_name["semantic_mismatch"]["evidence_type"] == "inferred_indicator"
    rankings["hybrid"] = hits(["b"])
    rankings["reranked"] = hits(["a"])
    assert "ranking_failure" in {
        i["category"]
        for i in classify_failures("alpha", rankings, {"a": 1}, hits(["b", "a"]), corpus, 1)
    }


@pytest.mark.parametrize("judgments", [{}, {"a": 0}])
def test_zero_judgments_retained_explicitly(judgments):
    corpus = CorpusFeatures([Document("a", "", "alpha")])
    rankings = {m: hits(["a"]) for m in METHODS}
    rows, analysis = query_comparison(
        "q",
        "alpha",
        rankings,
        judgments,
        [],
        {m: {"latency_ms": 1, "rerank_ms": 0} for m in METHODS},
        corpus,
        10,
    )
    assert len(rows) == 4
    assert all(r["ndcg@10"] == 0 for r in rows)
    assert all(r["judgment_status"] != "positive_judgments" for r in rows)
    assert any(i["category"] == "insufficient_judgments" for i in analysis["indicators"])


def test_split_exact_reproducible_and_disjoint():
    ids = [str(i) for i in range(300)]
    split = exploratory_split(ids)
    assert [len(split[n]) for n in ("development", "validation", "held_out")] == [180, 60, 60]
    assert split == exploratory_split(list(reversed(ids)))
    validate_partition(split, ids)
    assert split != exploratory_split(ids, 43)
    with pytest.raises(ValueError):
        exploratory_split(["a"] * 5)
    split["validation"].append(split["held_out"][0])
    with pytest.raises(ValueError):
        validate_partition(split, ids)


def test_router_rules_explanations_and_no_label_access():
    class System:
        def search(self, query, top_k=10):
            return hits(["a"])

    corpus = CorpusFeatures([Document("a", "", "rare"), Document("b", "", "common")])
    router = QueryAwareRouter(
        corpus, {"bm25": System(), "hybrid": System(), "reranked": System()}, RoutingPolicy(0.25, 0)
    )
    trace = router.search_with_trace("rare")
    assert trace.strategy == "bm25" and not trace.reranker_invoked
    assert "Rare-term proportion" in trace.explanation
    assert "judgments" not in vars(router) and "qrels" not in vars(router)
    assert trace.features == asdict(corpus.extract("rare"))
    assert router.search_with_trace("unknown").strategy == "hybrid"
    f = QueryFeatures(2, 2, 4, False, False, 0, 1)
    assert RoutingPolicy(1.01, 8).decide(f)[0] == "reranked"
    assert RoutingPolicy(1.01, 0).decide(f)[0] == "hybrid"
    assert RoutingPolicy().decide(corpus.extract(""))[0] == "bm25"
    with pytest.raises(ValueError):
        router.search(None)
    with pytest.raises(ValueError):
        router.search("rare", 0)


def test_tuning_prohibits_held_out_outcomes():
    ids = [str(i) for i in range(10)]
    split = exploratory_split(ids)
    features = {q: QueryFeatures(5, 5, 4, False, False, 0.25, 1) for q in ids}
    outcomes = {
        q: {m: {"ndcg": 0.5, "latency_ms": 10} for m in METHODS}
        for q in split["development"] + split["validation"]
    }
    selected = select_policy(features, outcomes, split)
    assert not set(selected["label_query_ids"]) & set(split["held_out"])
    contaminated = {**outcomes, split["held_out"][0]: outcomes[next(iter(outcomes))]}
    with pytest.raises(ValueError, match="held-out labels prohibited"):
        select_policy(features, contaminated, split)
    # No held-out ground-truth values can be passed through this public selection boundary.
    assert select_policy(features, dict(reversed(list(outcomes.items()))), split) == selected


def test_paired_bootstrap_known_difference():
    result = paired_bootstrap([2, 3, 4], [1, 2, 3], seed=42)
    assert result["mean_difference"] == result["ci95_lower"] == result["ci95_upper"] == 1
    assert result == paired_bootstrap([2, 3, 4], [1, 2, 3], seed=42)
    with pytest.raises(ValueError):
        paired_bootstrap([1], [])


@pytest.fixture
def offline_experiment(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(SMOKE, root)
    with (root / "queries.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"_id": "q5", "text": "Python virtual environments"}) + "\n")
        f.write(json.dumps({"_id": "q6", "text": "ERR_42"}) + "\n")
    with (root / "qrels/test.tsv").open("a") as f:
        f.write("q5\td1\t1\nq6\td2\t1\n")
    data = load_dataset(root)
    encoder, scorer = Encoder(), Scorer()
    source = run_comparison(
        root,
        tmp_path / "phase2",
        ComparisonConfig(k=3, candidate_depth=5, rerank_candidates=3, cache=None),
        encoder,
        scorer,
    )
    bm25 = BM25Retriever(data.documents.values())
    dense = DenseRetriever(data.documents.values(), encoder)
    hybrid = HybridRetriever(bm25, dense, 5)
    rerank = RerankingRetriever(hybrid, data.documents.values(), scorer, 3)
    systems = dict(zip(METHODS, (bm25, dense, hybrid, rerank)))
    return root, source, systems


def test_phase3_persistence_replay_and_source_preservation(offline_experiment, tmp_path):
    root, source, systems = offline_experiment
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    run = run_phase3(root, source, tmp_path / "phase3", systems_override=systems, plots=False)
    assert all((run / name).is_file() for name in REQUIRED)
    assert replay_phase3(run, root) == 10
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    meta = read_json(run / "run_metadata.json")
    assert meta["status"] == "complete"
    split = read_json(run / "split_metadata.json")
    policy = read_json(run / "frozen_policy.json")
    assert not set(split["held_out"]) & set(policy["tuning_label_query_ids"])
    (run / "frozen_policy.json").write_text("{}")
    with pytest.raises(ValueError, match="Artifact changed"):
        replay_phase3(run, root)
