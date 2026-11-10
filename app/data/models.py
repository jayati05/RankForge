"""Download revision-pinned local model snapshots using system HTTPS trust."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from app.retrieval.neural import model_identity

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def download_snapshot(model_name: str, destination: Path, revision: str | None = None) -> dict:
    destination = Path(destination)
    if (destination / "rankforge_model.json").exists():
        identity = model_identity(destination)
        if identity["model_name"] != model_name or (revision and identity["revision"] != revision):
            raise ValueError("Existing snapshot has another model/revision; use a new destination")
        return identity
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import truststore

    truststore.inject_into_ssl()  # Uses OS certificate trust; verification remains enabled.
    from huggingface_hub import HfApi, snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "download_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["model_name"] != model_name or (revision and revision != state["revision"]):
            raise ValueError("Partial download is for another model/revision")
        resolved = state["revision"]
    else:
        if any(destination.iterdir()):
            raise ValueError("Refusing an unrelated nonempty model destination")
        resolved = HfApi().model_info(model_name, revision=revision).sha
        state_path.write_text(
            json.dumps({"model_name": model_name, "revision": resolved}), encoding="utf-8"
        )
    print(f"Downloading {model_name}@{resolved} to {destination}", flush=True)
    snapshot_download(
        model_name,
        revision=resolved,
        local_dir=destination,
        max_workers=2,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "vocab.txt",
            "tokenizer.model",
            "README.md",
            "LICENSE*",
        ],
        ignore_patterns=["onnx/*", "openvino/*", "*.bin", "*.h5", "*.msgpack"],
    )
    files = {
        p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.rglob("*"))
        if p.is_file()
        and ".cache" not in p.relative_to(destination).parts
        and p.name != "download_state.json"
    }
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Model snapshot has no safetensors weights")
    manifest = {
        "model_name": model_name,
        "revision": resolved,
        "source": f"https://huggingface.co/{model_name}",
        "files_sha256": files,
        "license": "See included model card/license",
    }
    (destination / "rankforge_model.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return model_identity(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local sentence/cross encoder snapshots")
    parser.add_argument("--dense-model", default=DENSE_MODEL)
    parser.add_argument("--rerank-model", default=RERANK_MODEL)
    parser.add_argument("--dense-revision")
    parser.add_argument("--rerank-revision")
    parser.add_argument("--dense-path", type=Path, default=Path("models/dense"))
    parser.add_argument("--rerank-path", type=Path, default=Path("models/reranker"))
    args = parser.parse_args()
    try:
        for name, path, revision in [
            (args.dense_model, args.dense_path, args.dense_revision),
            (args.rerank_model, args.rerank_path, args.rerank_revision),
        ]:
            identity = download_snapshot(name, path, revision)
            print(f"Verified {identity['model_name']}@{identity['revision']}", flush=True)
    except Exception as exc:
        parser.exit(
            2,
            f"Model download failed: {type(exc).__name__}: {exc}\nRerun the same command to resume the pinned partial download.\n",
        )


if __name__ == "__main__":
    main()
