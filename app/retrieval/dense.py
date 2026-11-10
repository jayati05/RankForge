"""Normalized dense cosine retrieval with content-addressed cache validation."""

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import numpy as np

from app.data.schemas import Document
from app.retrieval.base import SearchResult, validate_top_k
from app.retrieval.neural import Encoder, document_text, positive_integer


class CacheError(ValueError):
    """An embedding cache does not match its corpus/model or is corrupt."""


def normalize(vectors: np.ndarray, rows: int, dimension: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (rows, dimension) or not np.isfinite(vectors).all():
        raise ValueError("Encoder returned wrong shape or nonfinite embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any() or not np.isfinite(norms).all():
        raise ValueError("Encoder returned zero or invalid embedding norms")
    return vectors / norms


class DenseRetriever:
    def __init__(
        self,
        documents: Iterable[Document],
        encoder: Encoder,
        batch_size: int = 16,
        cache_path: Path | None = None,
        require_cache: bool = False,
    ):
        positive_integer(batch_size, "batch_size")
        positive_integer(encoder.dimension, "embedding dimension")
        docs = sorted(documents, key=lambda d: d.id)
        if not docs or len({d.id for d in docs}) != len(docs):
            raise ValueError("Corpus must be nonempty with unique IDs")
        self.ids = [d.id for d in docs]
        self.encoder, self.batch_size = encoder, batch_size
        corpus_bytes = json.dumps(
            [[d.id, d.title, d.text] for d in docs], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.metadata = {
            "schema_version": 1,
            "model": encoder.identity,
            "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "document_ids": self.ids,
            "dimension": encoder.dimension,
            "preprocessing": "raw-title-space-text-v1",
            "dtype": "float32",
            "normalized": True,
        }
        self.cache_status = "disabled"
        if cache_path is not None and Path(cache_path).exists():
            self.vectors = self._load(Path(cache_path))
            self.cache_status = "hit"
        else:
            if require_cache:
                raise FileNotFoundError(
                    f"Dense cache missing: {cache_path}; build the dense index first"
                )
            start = perf_counter()
            self.vectors = normalize(
                encoder.encode([document_text(d.title, d.text) for d in docs], batch_size),
                len(docs),
                encoder.dimension,
            )
            self.original_build_seconds = perf_counter() - start
            if cache_path is not None:
                cache_path = Path(cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    **self.metadata,
                    "vectors_sha256": hashlib.sha256(self.vectors.tobytes()).hexdigest(),
                    "original_build_seconds": self.original_build_seconds,
                }
                with cache_path.open("xb") as stream:
                    np.savez_compressed(
                        stream, vectors=self.vectors, metadata=json.dumps(metadata, sort_keys=True)
                    )
                self.cache_status = "built"

    def _load(self, path: Path) -> np.ndarray:
        try:
            with np.load(path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata"].item()))
                vectors = cache["vectors"]
            checksum = metadata.pop("vectors_sha256")
            self.original_build_seconds = metadata.pop("original_build_seconds")
            if (
                not isinstance(self.original_build_seconds, (int, float))
                or not np.isfinite(self.original_build_seconds)
                or self.original_build_seconds < 0
            ):
                raise ValueError("invalid cached build time")
            if metadata != self.metadata:
                raise ValueError("model, corpus or preprocessing changed")
            if (
                vectors.dtype != np.float32
                or vectors.shape != (len(self.ids), self.encoder.dimension)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
                or hashlib.sha256(vectors.tobytes()).hexdigest() != checksum
            ):
                raise ValueError("invalid embedding contents")
            return vectors
        except Exception as exc:
            raise CacheError(
                f"Rejected dense cache {path}: {exc}. Use a new cache path to rebuild."
            ) from exc

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        validate_top_k(top_k)
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not query.strip():
            return []
        query_vector = normalize(
            self.encoder.encode([query], self.batch_size), 1, self.encoder.dimension
        )[0]
        scores = self.vectors @ query_vector
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))
        return [
            SearchResult(self.ids[i], rank, float(scores[i]))
            for rank, i in enumerate(order[:top_k], 1)
        ]


def main() -> None:
    from app.data.loaders import load_dataset
    from app.retrieval.neural import SentenceEncoder

    parser = argparse.ArgumentParser(description="Build or validate a dense index (CPU)")
    parser.add_argument("--dataset", type=Path, default=Path("data/public/scifact"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-path", type=Path, default=Path("models/dense"))
    parser.add_argument("--cache", type=Path, default=Path("indexes/scifact-minilm.npz"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        data = load_dataset(args.dataset, args.split)
        encoder = SentenceEncoder(args.model_path, threads=args.threads)
        retriever = DenseRetriever(data.documents.values(), encoder, args.batch_size, args.cache)
        print(
            json.dumps(
                {
                    "cache": str(args.cache),
                    "status": retriever.cache_status,
                    "documents": len(retriever.ids),
                    "dimension": encoder.dimension,
                }
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Dense indexing failed: {exc}\n")


if __name__ == "__main__":
    main()
