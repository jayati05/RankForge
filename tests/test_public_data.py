"""Offline BEIR-format tests use authored fixtures, never downloaded SciFact text."""

import csv
import hashlib
import io
import json
import shutil
import urllib.error
import zipfile
from pathlib import Path

import pytest

from app.data import download
from app.data.loaders import load_dataset
from app.data.selection import select_queries
from app.evaluation.benchmark import run_benchmark

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke"


@pytest.fixture
def archive(tmp_path, monkeypatch):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        for relative in download.MEMBERS:
            if relative == "qrels/train.tsv":
                content = "query-id\tcorpus-id\tscore\nq1\td2\t1\n"
            else:
                content = (SMOKE / relative).read_text(encoding="utf-8")
                if relative == "qrels/test.tsv":
                    content = content.replace("q1\td2\t1\n", "")
            bundle.writestr("scifact/" + relative, content)
        bundle.writestr("../../escape.txt", "never extract this")
    # Production checksum is never relaxed; local fixture substitutes its own pinned digest.
    monkeypatch.setattr(download, "SCIFACT_MD5", hashlib.md5(path.read_bytes()).hexdigest())
    return path


def test_import_preserves_bytes_and_provenance(archive, tmp_path):
    target = tmp_path / "dataset"
    download.import_archive(archive, target)
    with zipfile.ZipFile(archive) as bundle:
        for relative in download.MEMBERS:
            assert (target / relative).read_bytes() == bundle.read("scifact/" + relative)
    metadata = json.loads((target / "metadata.json").read_text())
    assert metadata["source_archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert metadata["license"] == "CC-BY-SA-4.0"
    assert metadata["splits"]["test"] == {"queries": 3, "judgments": 3}
    assert not (tmp_path / "escape.txt").exists()
    assert set(load_dataset(target, "train").queries) == {"q1"}
    assert set(load_dataset(target, "test").queries) == {"q2", "q3", "q4"}
    assert download.download_scifact(target) == target  # no network on verified reuse
    with pytest.raises(FileExistsError):
        download.import_archive(archive, target)
    (target / "corpus.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        download.download_scifact(target)


def test_checksum_rejection_leaves_no_dataset(archive, tmp_path):
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    target = tmp_path / "dataset"
    with pytest.raises(ValueError, match="MD5 mismatch"):
        download.import_archive(archive, target)
    assert not target.exists()


def test_incomplete_archive_leaves_no_dataset(tmp_path, monkeypatch):
    archive = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("scifact/corpus.jsonl", "{}")
    monkeypatch.setattr(download, "SCIFACT_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="requires exactly one"):
        download.import_archive(archive, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob("scifact-stage-*"))


def test_network_download_offline(archive, tmp_path, monkeypatch):
    def fake_open(request, timeout):
        assert request.full_url == download.SCIFACT_URL
        assert timeout == 60
        return io.BytesIO(archive.read_bytes())

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_open)
    target = download.download_scifact(tmp_path / "dataset")
    assert len(load_dataset(target).queries) == 3
    assert not list(tmp_path.glob("scifact-download-*"))


def test_network_failure_cleanup(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(download.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.URLError):
        download.download_scifact(tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob("scifact-download-*"))


def test_subset_preserves_corpus_and_complete_judgments():
    data = load_dataset(SMOKE)
    data.qrels["q1"]["d6"] = 1
    subset = select_queries(data, 2, 42)
    expected = sorted(
        sorted(data.queries, key=lambda q: hashlib.sha256(f"42:{q}".encode()).hexdigest())[:2]
    )
    assert list(subset.queries) == expected
    assert subset.documents is data.documents
    assert len(subset.documents) == 6
    assert subset.qrels == {qid: data.qrels[qid] for qid in expected}
    assert list(select_queries(data, 2, 42).queries) == expected
    reversed_data = load_dataset(SMOKE)
    reversed_data.queries = dict(reversed(list(reversed_data.queries.items())))
    assert list(select_queries(reversed_data, 2, 42).queries) == expected
    assert len(data.queries) == 4
    assert set(select_queries(data, 100).queries) == set(data.queries)
    assert set(select_queries(data).queries) == set(data.queries)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_subset_invalid_limit(limit):
    with pytest.raises(ValueError):
        select_queries(load_dataset(SMOKE), limit)


def test_string_ids_and_multirelevant_join(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id":"001","text":"alpha"}\n{"_id":"1","title":"beta","text":"gamma"}\n'
    )
    (tmp_path / "queries.jsonl").write_text(
        '{"_id":"0007","text":"alpha"}\n{"_id":"7","text":"gamma"}\n'
    )
    (tmp_path / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n0007\t001\t2\n0007\t1\t1\n"
    )
    data = load_dataset(tmp_path)
    assert set(data.documents) == {"001", "1"}
    assert list(data.queries) == ["0007"]
    assert data.qrels == {"0007": {"001": 2, "1": 1}}
    assert data.documents["001"].title == ""


@pytest.mark.parametrize("content", ["[]\n", "null\n", "42\n"])
def test_nonobject_jsonl_error(tmp_path, content):
    shutil.copytree(SMOKE, tmp_path / "data")
    (tmp_path / "data/corpus.jsonl").write_text(content)
    with pytest.raises(ValueError, match="records must be objects"):
        load_dataset(tmp_path / "data")


@pytest.mark.parametrize("row", ["q1\td2\t1\textra", "q1\td2"])
def test_invalid_tsv_width(tmp_path, row):
    shutil.copytree(SMOKE, tmp_path / "data")
    (tmp_path / "data/qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\n" + row + "\n")
    with pytest.raises(ValueError, match="exactly three"):
        load_dataset(tmp_path / "data")


def test_subset_benchmark_matches_full_ranking_and_manual_scores(tmp_path):
    full = run_benchmark(SMOKE, tmp_path / "runs")
    subset = run_benchmark(SMOKE, tmp_path / "runs", max_queries=2, seed=42)
    full_rankings = json.loads((full / "rankings.json").read_text())
    subset_rankings = json.loads((subset / "rankings.json").read_text())
    assert subset_rankings == {qid: full_rankings[qid] for qid in subset_rankings}
    metadata = json.loads((subset / "run_metadata.json").read_text())
    assert metadata["document_count"] == 6
    assert metadata["query_count"] == 2
    assert metadata["available_query_count"] == 4
    assert metadata["subset"]["seed"] == 42
    assert metadata["evaluation"]["unjudged_gain"] == 0
    assert "code_sha256" in metadata
    with (subset / "per_query_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["query_id"] for row in rows} == set(subset_rankings)
    for row in rows:
        # Authored smoke cases each have a single positive at rank 1.
        assert float(row["recall@10"]) == 1
        assert float(row["precision@10"]) == 0.1
        assert float(row["mrr@10"]) == 1
        assert float(row["ndcg@10"]) == 1
