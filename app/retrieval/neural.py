"""Local-only model adapters. Imports remain optional for lexical usage and tests."""

import hashlib
import json
from pathlib import Path
from importlib.metadata import version
from typing import Protocol

import numpy as np


class ModelError(RuntimeError):
    """A neural model could not be initialized or used."""


class Encoder(Protocol):
    identity: dict
    dimension: int

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray: ...


class PairScorer(Protocol):
    identity: dict

    def score(self, pairs: list[tuple[str, str]], batch_size: int) -> np.ndarray: ...


def document_text(title: str, text: str) -> str:
    return title + " " + text


def positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def model_identity(path: str | Path) -> dict:
    path = Path(path)
    try:
        provenance = json.loads((path / "rankforge_model.json").read_text(encoding="utf-8"))
        files = provenance["files_sha256"]
        if not files:
            raise ValueError("Empty model manifest")
        for name, expected in files.items():
            target = (path / name).resolve()
            if not target.is_relative_to(path.resolve()) or not target.is_file():
                raise ValueError("Invalid model manifest path")
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Model file changed: {name}")
        return {
            "model_name": provenance["model_name"],
            "revision": provenance["revision"],
            "files_sha256": files,
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelError(
            f"Invalid/missing model snapshot at {path}; run python -m app.data.models: {exc}"
        ) from exc


def configure_torch(threads: int, seed: int) -> None:
    positive_integer(threads, "threads")
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class SentenceEncoder:
    def __init__(self, path: str | Path, threads: int = 4, seed: int = 42, max_length: int = 256):
        positive_integer(max_length, "max_length")
        self.identity = model_identity(path)
        self.identity["library_versions"] = {
            name: version(name)
            for name in ("sentence-transformers", "transformers", "torch", "numpy")
        }
        try:
            configure_torch(threads, seed)
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                str(path), device="cpu", local_files_only=True, trust_remote_code=False
            )
            self.model.max_seq_length = max_length
            self.model.eval()
            self.dimension = int(self.model.get_sentence_embedding_dimension())
            self.identity.update(
                {
                    "max_length": max_length,
                    "normalization": "l2-float32",
                    "input": "raw-title-space-text-v1",
                }
            )
        except Exception as exc:
            raise ModelError(f"Could not load sentence encoder {path}: {exc}") from exc

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        try:
            return np.asarray(
                self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            raise ModelError(f"Sentence encoder inference failed: {exc}") from exc


class CrossEncoderScorer:
    def __init__(self, path: str | Path, threads: int = 4, seed: int = 42, max_length: int = 512):
        positive_integer(max_length, "max_length")
        self.identity = model_identity(path)
        self.identity["library_versions"] = {
            name: version(name)
            for name in ("sentence-transformers", "transformers", "torch", "numpy")
        }
        try:
            configure_torch(threads, seed)
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                str(path),
                device="cpu",
                max_length=max_length,
                local_files_only=True,
                trust_remote_code=False,
            )
            self.model.model.eval()
            self.identity.update(
                {
                    "max_length": max_length,
                    "input": "query + raw-title-space-text-v1",
                    "activation": "Identity (raw logits)",
                }
            )
        except Exception as exc:
            raise ModelError(f"Could not load cross-encoder {path}: {exc}") from exc

    def score(self, pairs: list[tuple[str, str]], batch_size: int) -> np.ndarray:
        try:
            import torch

            return np.asarray(
                self.model.predict(
                    pairs,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    activation_fn=torch.nn.Identity(),
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            raise ModelError(f"Cross-encoder inference failed: {exc}") from exc
