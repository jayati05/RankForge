import json
import math
import shutil
from pathlib import Path

import pytest

from app.data.loaders import load_dataset
from app.data.preprocessing import tokenize
from app.data.schemas import Document, Query
from app.evaluation.benchmark import run_benchmark
from app.evaluation.metrics import ranking_metrics
from app.retrieval.bm25 import BM25Retriever

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke"


@pytest.mark.parametrize(
    "query,expected", [("ERR_42", "d2"), ("BM25", "d3"), ("C++ pointers", "d5"), ("Python", "d1")]
)
def test_exact_and_rare_terms(query, expected):
    data = load_dataset(SMOKE)
    results = BM25Retriever(data.documents.values()).search(query, 3)
    assert results[0].document_id == expected
    assert [r.rank for r in results] == [1, 2, 3]
    assert all(math.isfinite(r.score) for r in results)


def test_preprocessing():
    assert tokenize("ERR_42 C++ Python3.11 caf\u00e9") == [
        "err_42",
        "c++",
        "python3.11",
        "caf\u00e9",
    ]
    assert tokenize("  !!! ") == []
    assert tokenize("\uff30\uff39\uff34\uff28\uff2f\uff2e") == ["python"]


@pytest.mark.parametrize("query", ["", "   ", "!!!", "unknownzzz"])
def test_empty_or_unseen(query):
    assert BM25Retriever(load_dataset(SMOKE).documents.values()).search(query) == []


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        BM25Retriever(load_dataset(SMOKE).documents.values()).search("Python", k)
    with pytest.raises(ValueError):
        ranking_metrics([], {}, k)


def test_invalid_corpus_and_parameters():
    for docs in [[], [Document("a", "", "!!!")], [Document("a", "", "x"), Document("a", "", "y")]]:
        with pytest.raises(ValueError):
            BM25Retriever(docs)
    for args in [{"k1": 0}, {"b": 2}, {"k1": float("nan")}]:
        with pytest.raises(ValueError):
            BM25Retriever(load_dataset(SMOKE).documents.values(), **args)


def test_ties_and_short_corpus():
    retriever = BM25Retriever([Document("b", "", "x"), Document("a", "", "x")])
    assert [r.document_id for r in retriever.search("x", 10)] == ["a", "b"]
    assert len(BM25Retriever([Document("a", "", "x"), Document("b", "", "")]).search("x")) == 2


def test_manual_metrics():
    values = ranking_metrics(["unjudged", "b", "a"], {"a": 3, "b": 1, "c": 2}, 3)
    expected = (1 / math.log2(3) + 7 / 2) / (7 + 3 / math.log2(3) + 1 / 2)
    assert values == pytest.approx(
        {"recall@3": 2 / 3, "precision@3": 2 / 3, "mrr@3": 1 / 2, "ndcg@3": expected}
    )
    assert ranking_metrics(["a"], {"a": 1}, 10)["precision@10"] == 0.1
    assert all(v == 0 for v in ranking_metrics([], {}, 10).values())
    assert all(v == 0 for v in ranking_metrics(["a"], {"a": 0}, 10).values())
    assert ranking_metrics(["a"], {"a": 1}, 1)["ndcg@1"] == 1


def test_metric_rejects_duplicates_and_bad_grades():
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], {"a": 1}, 1)
    for grade in [-1, 1.5, float("nan"), True]:
        with pytest.raises(ValueError):
            ranking_metrics([], {"a": grade}, 1)


@pytest.mark.parametrize(
    "factory,args", [(Document, ("", "", "x")), (Document, ("a", None, "x")), (Query, ("q", " "))]
)
def test_schema_validation(factory, args):
    with pytest.raises(ValueError):
        factory(*args)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_doc",
        "missing_doc",
        "missing_query",
        "duplicate_qrel",
        "negative",
        "bad_header",
        "malformed",
    ],
)
def test_loader_rejects_invalid_data(tmp_path, mutation):
    root = tmp_path / "data"
    shutil.copytree(SMOKE, root)
    corpus = root / "corpus.jsonl"
    qrels = root / "qrels/test.tsv"
    if mutation == "duplicate_doc":
        corpus.write_text(corpus.read_text() + corpus.read_text().splitlines()[0] + "\n")
    elif mutation == "malformed":
        corpus.write_text("{broken\n")
    else:
        text = qrels.read_text()
        text = {
            "missing_doc": text.replace("d2", "absent"),
            "missing_query": text.replace("q1", "absent"),
            "duplicate_qrel": text + "q1\td2\t1\n",
            "negative": text.replace("q1\td2\t1", "q1\td2\t-1"),
            "bad_header": text.replace("query-id", "query"),
        }[mutation]
        qrels.write_text(text)
    with pytest.raises(ValueError):
        load_dataset(root)


def test_split_selection_and_path_validation(tmp_path):
    shutil.copytree(SMOKE, tmp_path / "data")
    root = tmp_path / "data"
    (root / "qrels/dev.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td2\t1\n")
    assert list(load_dataset(root, "dev").queries) == ["q1"]
    with pytest.raises(ValueError):
        load_dataset(root, "../test")
    with pytest.raises(FileNotFoundError):
        load_dataset(root, "missing")


def test_benchmark_end_to_end(tmp_path):
    import csv

    first = run_benchmark(SMOKE, tmp_path)
    second = run_benchmark(SMOKE, tmp_path)
    assert first != second
    metadata = json.loads((first / "run_metadata.json").read_text())
    assert metadata["dataset"]["synthetic"] is True
    assert metadata["query_ids"] == ["q1", "q2", "q3", "q4"]
    assert len(metadata["source_sha256"]) == 4
    rankings = json.loads((first / "rankings.json").read_text())
    with (first / "per_query_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["query_id"] for row in rows} == set(rankings)
    for row in rows:
        ids = [r["document_id"] for r in rankings[row["query_id"]]]
        assert len(ids) == len(set(ids))
        expected = ranking_metrics(ids, load_dataset(SMOKE).qrels[row["query_id"]], 10)
        for key, value in expected.items():
            assert float(row[key]) == pytest.approx(value)
        assert all(math.isfinite(float(v)) for key, v in row.items() if key != "query_id")
    assert (first / "benchmark_summary.csv").is_file()
