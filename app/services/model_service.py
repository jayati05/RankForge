"""Per-process lazy resources, bounded serialized search and cheap health snapshots."""

import importlib.util
import json
import re
import threading
from contextlib import contextmanager
from time import perf_counter
from typing import Callable

from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.logging import log_event
from app.data.loaders import read_jsonl
from app.data.schemas import Document
from app.evaluation.features import CorpusFeatures
from app.evaluation.transfer_policy import load_protocol
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.neural import CrossEncoderScorer, SentenceEncoder
from app.retrieval.reranker import RerankingRetriever
from app.retrieval.router import QueryAwareRouter

METHODS = ("bm25", "dense", "hybrid", "reranked", "router")
RESOURCES = (
    "corpus",
    "bm25",
    "encoder",
    "dense",
    "hybrid",
    "scorer",
    "reranked",
    "features",
    "policy",
    "router",
)
MESSAGES = {
    "corpus": "Configured corpus is unavailable or invalid; prepare the dataset and restart.",
    "encoder": "Dense model is unavailable; verify the local snapshot and neural dependencies, then restart.",
    "dense": "Dense retrieval is unavailable; verify the prepared index, local model and neural dependencies, then restart.",
    "scorer": "Reranking model is unavailable; verify the local snapshot and neural dependencies, then restart.",
    "policy": "Frozen routing policy is unavailable or incompatible; verify the saved transfer capsule and restart.",
    "reranked": "Reranking setup is unavailable; check candidate limits and prepared corpus, then restart.",
}


class LazyRetriever:
    def __init__(self, manager, method: str):
        self.manager, self.method = manager, method

    def search(self, query: str, top_k: int = 10):
        return self.manager.get(self.method).search(query, top_k)

    def search_with_trace(self, query: str, top_k: int = 10):
        return self.manager.get(self.method).search_with_trace(query, top_k)


class ModelService:
    def __init__(
        self,
        settings: Settings,
        encoder_factory: Callable = SentenceEncoder,
        scorer_factory: Callable = CrossEncoderScorer,
    ):
        self.settings = settings
        self.encoder_factory, self.scorer_factory = encoder_factory, scorer_factory
        self._init_lock = threading.RLock()
        self._status_lock = threading.Lock()
        self._work_lock = threading.Lock()
        self._admission = threading.BoundedSemaphore(settings.max_queued_requests + 1)
        self._resources = {}
        self._states = {
            name: {
                "state": "not_loaded",
                "initialization_ms": 0.0,
                "error_code": None,
                "initialization_attempts": 0,
                "initialization_count": 0,
            }
            for name in RESOURCES
        }
        self._depth = 0
        self.initialization_seconds = 0.0
        self.initialized_names = []
        self._closed = False
        self._stopping = threading.Event()
        self._active = self._queued = 0

    @contextmanager
    def execution(self):
        start = perf_counter()
        if self._stopping.is_set():
            raise ServiceError("service_stopping", "Service is shutting down.")
        if not self._admission.acquire(blocking=False):
            raise ServiceError("server_busy", "Search capacity is full; retry later.", 429)
        acquired = False
        waiting = True
        active = False
        with self._status_lock:
            self._queued += 1
        try:
            acquired = self._work_lock.acquire(timeout=self.settings.max_queue_wait_seconds)
            if not acquired:
                raise ServiceError("server_busy", "Search queue wait exceeded; retry later.", 429)
            with self._status_lock:
                self._queued -= 1
                waiting = False
            if self._stopping.is_set():
                raise ServiceError("service_stopping", "Service is shutting down.")
            with self._status_lock:
                self._active += 1
                active = True
            yield (perf_counter() - start) * 1000
        finally:
            with self._status_lock:
                if waiting:
                    self._queued -= 1
                if active:
                    self._active -= 1
            if acquired:
                self._work_lock.release()
            self._admission.release()

    def _state(self, name: str, **values) -> None:
        with self._status_lock:
            self._states[name].update(values)

    def get(self, name: str):
        with self._init_lock:
            if self._closed:
                raise ServiceError("service_stopping", "Service is shutting down.")
            if name in self._resources:
                return self._resources[name]
            if name not in self._states:
                raise ValueError("Unknown internal resource")
            if self._states[name]["state"] == "failed":
                raise ServiceError(
                    name + "_unavailable",
                    MESSAGES.get(
                        name,
                        "A required resource is unavailable; check service status and restart.",
                    ),
                )
            self._depth += 1
            outer = self._depth == 1
            start = perf_counter()
            self._state(
                name,
                state="initializing",
                initialization_attempts=self._states[name]["initialization_attempts"] + 1,
            )
            try:
                resource = self._build(name)
                self._resources[name] = resource
                self.initialized_names.append(name)
                self._state(
                    name,
                    state="ready",
                    initialization_ms=(perf_counter() - start) * 1000,
                    initialization_count=self._states[name]["initialization_count"] + 1,
                )
                log_event(
                    "resource_initialized",
                    resource=name,
                    duration_ms=round((perf_counter() - start) * 1000, 3),
                )
                return resource
            except Exception as exc:
                self._state(
                    name,
                    state="failed",
                    error_code=name + "_unavailable",
                    initialization_ms=(perf_counter() - start) * 1000,
                )
                log_event(
                    "resource_initialization_failed",
                    resource=name,
                    exception_type=type(exc).__name__,
                )
                if isinstance(exc, ServiceError):
                    raise
                raise ServiceError(
                    name + "_unavailable",
                    MESSAGES.get(
                        name,
                        "A required resource could not be initialized; inspect status and restart.",
                    ),
                ) from exc
            finally:
                if outer:
                    self.initialization_seconds += perf_counter() - start
                self._depth -= 1

    def _build(self, name: str):
        s = self.settings
        if name == "corpus":
            return read_jsonl(s.dataset_dir / "corpus.jsonl", Document)
        if name == "bm25":
            return BM25Retriever(self.get("corpus").values())
        if name == "encoder":
            return self.encoder_factory(s.dense_model_dir, threads=s.threads, seed=42)
        if name == "dense":
            if not s.dense_index.is_file():
                raise FileNotFoundError("Prepared dense cache required")
            return DenseRetriever(
                self.get("corpus").values(), self.get("encoder"), s.batch_size, s.dense_index, True
            )
        if name == "hybrid":
            return HybridRetriever(self.get("bm25"), self.get("dense"), s.candidate_depth, 60)
        if name == "scorer":
            return self.scorer_factory(s.rerank_model_dir, threads=s.threads, seed=42)
        if name == "reranked":
            return RerankingRetriever(
                self.get("hybrid"),
                self.get("corpus").values(),
                self.get("scorer"),
                s.rerank_candidates,
                s.batch_size,
            )
        if name == "features":
            return CorpusFeatures(self.get("corpus").values())
        if name == "policy":
            return load_protocol(s.policy_dir)
        if name == "router":
            _, policy = self.get("policy")
            return QueryAwareRouter(
                self.get("features"),
                {m: LazyRetriever(self, m) for m in ("bm25", "hybrid", "reranked")},
                policy,
            )
        raise ValueError("Unknown internal resource")

    def snapshot(self) -> dict:
        with self._status_lock:
            states = {k: dict(v) for k, v in self._states.items()}
            closed = self._closed
            activity = {
                "active": self._active,
                "queued": self._queued,
                "stopping": self._stopping.is_set(),
            }
        s = self.settings
        present = {
            "corpus": (s.dataset_dir / "corpus.jsonl").is_file(),
            "dense_index": s.dense_index.is_file(),
            "dense_model": (s.dense_model_dir / "rankforge_model.json").is_file(),
            "rerank_model": (s.rerank_model_dir / "rankforge_model.json").is_file(),
            "policy": all(
                (s.policy_dir / p).is_file() for p in ("protocol.json", "frozen_policy.json")
            ),
        }
        neural = all(
            importlib.util.find_spec(name) is not None
            for name in ("sentence_transformers", "torch")
        )
        possible = {
            "bm25": present["corpus"],
            "dense": present["corpus"]
            and present["dense_index"]
            and present["dense_model"]
            and neural,
        }
        possible["hybrid"] = possible["bm25"] and possible["dense"]
        possible["reranked"] = possible["hybrid"] and present["rerank_model"]
        possible["router"] = possible["hybrid"] and present["policy"]
        dependency_names = {
            "bm25": ("corpus", "bm25"),
            "dense": ("corpus", "encoder", "dense"),
            "hybrid": ("corpus", "bm25", "encoder", "dense", "hybrid"),
            "reranked": ("corpus", "bm25", "encoder", "dense", "hybrid", "scorer", "reranked"),
            "router": (
                "corpus",
                "bm25",
                "encoder",
                "dense",
                "hybrid",
                "features",
                "policy",
                "router",
            ),
        }
        for method, names in dependency_names.items():
            possible[method] = possible[method] and not any(
                states[name]["state"] == "failed" for name in names
            )
        available = [
            m
            for m in METHODS
            if possible[m] and states[m]["state"] != "failed" and not activity["stopping"]
        ]
        return {
            "status": "stopped"
            if closed
            else "stopping"
            if activity["stopping"]
            else "degraded"
            if not present["corpus"] or any(v["state"] == "failed" for v in states.values())
            else "ok",
            "dataset": {
                "name": s.dataset_label,
                "available": present["corpus"],
                "loaded": states["corpus"]["state"] == "ready",
            },
            "indexes": {
                "bm25": {"state": states["bm25"]["state"]},
                "dense": {"present": present["dense_index"], "state": states["dense"]["state"]},
            },
            "resources": states,
            "activity": activity,
            "available_methods": available,
            "configuration_present": present,
            "neural_dependencies_available": neural,
            "availability_note": "File/dependency presence is provisional until first use validates resources; status checks do not initialize them.",
        }

    def models(self) -> dict:
        snapshot = self.snapshot()
        result = {}
        for role, key, directory in (
            ("dense", "encoder", self.settings.dense_model_dir),
            ("reranker", "scorer", self.settings.rerank_model_dir),
        ):
            identifier, revision = None, None
            manifest = directory / "rankforge_model.json"
            try:
                if manifest.is_file() and manifest.stat().st_size <= 65536:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    name = data.get("model_name", "")
                    identifier = (
                        name
                        if isinstance(name, str)
                        and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name)
                        else "local-model"
                    )
                    rev = data.get("revision", "")
                    revision = (
                        rev
                        if isinstance(rev, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", rev)
                        else None
                    )
            except (OSError, ValueError, AttributeError):
                pass
            result[role] = {
                "identifier": identifier,
                "revision": revision,
                "device": "cpu",
                "snapshot_present": manifest.is_file(),
                **snapshot["resources"][key],
            }
        return {
            "models": result,
            "indexes": snapshot["indexes"],
            "neural_dependencies_available": snapshot["neural_dependencies_available"],
        }

    def close(self) -> None:
        # Reject new/queued searches immediately, while letting active work finish safely.
        self._stopping.set()
        with self._work_lock, self._init_lock:
            self._closed = True
            self._resources.clear()
            with self._status_lock:
                for state in self._states.values():
                    if state["state"] == "ready":
                        state["state"] = "released"
            log_event("resources_released")
