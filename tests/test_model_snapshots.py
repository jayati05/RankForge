import hashlib
import json
import sys
import types

import pytest

from app.data.models import download_snapshot
from app.retrieval import neural
from app.retrieval.neural import ModelError, model_identity


def snapshot(tmp_path):
    (tmp_path / "weights.test").write_bytes(b"authored fake weights")
    provenance = {
        "model_name": "offline/test",
        "revision": "a" * 40,
        "files_sha256": {"weights.test": hashlib.sha256(b"authored fake weights").hexdigest()},
    }
    (tmp_path / "rankforge_model.json").write_text(json.dumps(provenance))
    return provenance


def test_model_snapshot_reuse_and_invalidation(tmp_path):
    expected = snapshot(tmp_path)
    assert model_identity(tmp_path) == expected
    assert download_snapshot("offline/test", tmp_path, "a" * 40) == expected
    with pytest.raises(ValueError, match="another model/revision"):
        download_snapshot("offline/test", tmp_path, "b" * 40)
    (tmp_path / "weights.test").write_bytes(b"modified")
    with pytest.raises(ModelError, match="Model file changed"):
        model_identity(tmp_path)


def test_model_manifest_path_escape_rejected(tmp_path):
    provenance = snapshot(tmp_path)
    provenance["files_sha256"] = {"../outside": "a" * 64}
    (tmp_path / "rankforge_model.json").write_text(json.dumps(provenance))
    with pytest.raises(ModelError, match="Invalid model manifest path"):
        model_identity(tmp_path)


@pytest.mark.parametrize(
    "adapter,name",
    [(neural.SentenceEncoder, "sentence encoder"), (neural.CrossEncoderScorer, "cross-encoder")],
)
def test_model_load_failures_are_clear_and_offline(tmp_path, monkeypatch, adapter, name):
    snapshot(tmp_path)
    monkeypatch.setattr(neural, "configure_torch", lambda *args: None)
    monkeypatch.setattr(neural, "version", lambda _: "test")

    def fail(*args, **kwargs):
        raise RuntimeError("controlled model loader failure")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=fail, CrossEncoder=fail),
    )
    with pytest.raises(ModelError, match="Could not load " + name):
        adapter(tmp_path)
