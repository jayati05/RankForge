import json

import numpy as np
import pytest

from app.data.schemas import Document
from app.retrieval.dense import CacheError, DenseRetriever
from app.retrieval.neural import ModelError, model_identity


class FakeEncoder:
    dimension = 2
    identity = {"model_name": "controlled-test", "revision": "v1"}

    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size):
        self.calls.append((texts, batch_size))
        values = {" A": [2, 0], " B": [1, 1], " C": [0, 3], "q": [5, 0]}
        return np.array([values[text] for text in texts], dtype=np.float32)


@pytest.fixture
def docs():
    return [Document("c", "", "C"), Document("b", "", "B"), Document("a", "", "A")]


def test_cosine_and_precomputation(docs):
    encoder = FakeEncoder()
    retriever = DenseRetriever(docs, encoder, batch_size=2)
    results = retriever.search("q", 10)
    assert [r.document_id for r in results] == ["a", "b", "c"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.score for r in results] == pytest.approx([1, 1 / np.sqrt(2), 0])
    retriever.search("q")
    assert encoder.calls == [([" A", " B", " C"], 2), (["q"], 2), (["q"], 2)]
    assert retriever.search(" ") == []


def test_dense_ties(docs):
    docs = [Document("b", "", "A"), Document("a", "", "A")]
    assert [r.document_id for r in DenseRetriever(docs, FakeEncoder()).search("q")] == ["a", "b"]


def test_cache_hit_does_not_encode_corpus(docs, tmp_path):
    cache = tmp_path / "index.npz"
    first = DenseRetriever(docs, FakeEncoder(), cache_path=cache)
    encoder = FakeEncoder()
    second = DenseRetriever(docs, encoder, cache_path=cache)
    assert first.cache_status == "built" and second.cache_status == "hit"
    assert not encoder.calls
    assert second.search("q") == first.search("q")


@pytest.mark.parametrize(
    "change", ["text", "ids", "revision", "dimension", "preprocessing", "contents"]
)
def test_cache_rejects_stale_or_corrupt(docs, tmp_path, change):
    cache = tmp_path / "index.npz"
    DenseRetriever(docs, FakeEncoder(), cache_path=cache)
    encoder = FakeEncoder()
    if change == "text":
        docs[0] = Document("c", "", "changed")
    elif change == "ids":
        docs[0] = Document("d", "", "C")
    elif change == "revision":
        encoder.identity = {**encoder.identity, "revision": "v2"}
    elif change == "dimension":
        encoder.dimension = 3
    else:
        with np.load(cache, allow_pickle=False) as data:
            vectors = data["vectors"].copy()
            metadata = json.loads(data["metadata"].item())
        if change == "preprocessing":
            metadata["preprocessing"] = "changed"
        else:
            vectors[0, 0] = float("nan")
        np.savez(cache, vectors=vectors, metadata=json.dumps(metadata))
    with pytest.raises(CacheError, match="Rejected dense cache"):
        DenseRetriever(docs, encoder, cache_path=cache)


@pytest.mark.parametrize("value", [0, -1, True, 1.2])
def test_dense_invalid_values(docs, value):
    with pytest.raises(ValueError):
        DenseRetriever(docs, FakeEncoder(), batch_size=value)
    with pytest.raises(ValueError):
        DenseRetriever(docs, FakeEncoder()).search("q", value)


@pytest.mark.parametrize("vectors", [np.zeros((3, 2)), np.ones((2, 2)), np.full((3, 2), np.nan)])
def test_dense_invalid_embeddings(docs, vectors):
    encoder = FakeEncoder()
    encoder.encode = lambda *args: vectors
    with pytest.raises(ValueError):
        DenseRetriever(docs, encoder)


def test_dense_missing_or_invalid_index(docs, tmp_path):
    with pytest.raises(FileNotFoundError, match="Dense cache missing"):
        DenseRetriever(docs, FakeEncoder(), cache_path=tmp_path / "absent", require_cache=True)
    with pytest.raises(ValueError):
        DenseRetriever([], FakeEncoder())
    with pytest.raises(ValueError):
        DenseRetriever([docs[0], docs[0]], FakeEncoder())
    with pytest.raises(ValueError):
        DenseRetriever(docs, FakeEncoder()).search(None)
    with pytest.raises(ModelError, match="Invalid/missing model snapshot"):
        model_identity(tmp_path)
