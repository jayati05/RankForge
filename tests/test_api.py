"""Offline service/API tests use authored corpus, prepared vectors and model doubles."""

import asyncio
import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import sleep

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.main import create_app
from app.api.middleware import BoundaryMiddleware
from app.api.schemas import CompareResponse, SearchResponse
from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.logging import JsonFormatter, logger
from app.data.loaders import read_jsonl
from app.data.schemas import Document
from app.evaluation.routing_experiment import freeze_digest
from app.evaluation.transfer_policy import POLICY_FILES, freeze_transfer
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.router import RoutingPolicy
from app.services.comparison_service import ComparisonService
from app.services.model_service import ModelService
from app.services.search_service import SearchService

SMOKE = Path(__file__).resolve().parents[1] / "data/smoke"


class Encoder:
    dimension = 4
    identity = {"model_name": "unit/dense", "revision": "unit-v1"}

    def encode(self, texts, batch_size):
        return np.array([[1 + b for b in hashlib.sha256(t.encode()).digest()[:4]] for t in texts])


class Scorer:
    identity = {"model_name": "unit/scorer", "revision": "unit-v1"}

    def score(self, pairs, batch_size):
        return np.array([hashlib.sha256((q + t).encode()).digest()[0] for q, t in pairs])


@pytest.fixture(autouse=True)
def isolated_log_handlers():
    saved = list(logger.handlers)
    logger.handlers = []
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.handlers = saved


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    policy = {
        "policy": asdict(RoutingPolicy(1.01, 0)),
        "fixture": "authored offline service policy",
    }
    (source / "frozen_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    workspace = SMOKE.parents[1]
    metadata = {
        "status": "complete",
        "run_id": "authored-policy-fixture",
        "frozen_policy_sha256": freeze_digest(policy),
        "pipeline_config": {},
        "code_sha256": {
            p: hashlib.sha256((workspace / p).read_bytes()).hexdigest() for p in POLICY_FILES
        },
    }
    (source / "run_metadata.json").write_text(json.dumps(metadata))
    policy_dir = freeze_transfer(source, tmp_path / "policy")
    model_dirs = []
    for name in ("dense", "reranker"):
        path = tmp_path / name
        path.mkdir()
        (path / "rankforge_model.json").write_text(
            json.dumps({"model_name": "unit/" + name, "revision": "unit-v1"})
        )
        model_dirs.append(path)
    index = tmp_path / "prepared.npz"
    docs = read_jsonl(SMOKE / "corpus.jsonl", Document)
    DenseRetriever(docs.values(), Encoder(), cache_path=index)
    settings = Settings(
        dataset_dir=SMOKE,
        dataset_label="synthetic-smoke",
        dense_model_dir=model_dirs[0],
        rerank_model_dir=model_dirs[1],
        dense_index=index,
        policy_dir=policy_dir,
        default_top_k=3,
        candidate_depth=5,
        rerank_candidates=3,
    )
    calls = {"encoder": 0, "scorer": 0}

    def encoder(*args, **kwargs):
        calls["encoder"] += 1
        return Encoder()

    def scorer(*args, **kwargs):
        calls["scorer"] += 1
        return Scorer()

    manager = ModelService(settings, encoder, scorer)
    return settings, manager, calls


@pytest.fixture
def client(setup):
    settings, manager, _ = setup
    with TestClient(create_app(settings, lambda _: manager)) as http:
        yield http


def test_health_methods_models_docs_are_cold(client, setup, monkeypatch):
    _, manager, calls = setup

    def forbidden(*args, **kwargs):
        raise AssertionError("Status must not initialize anything")

    monkeypatch.setattr(manager, "_build", forbidden)
    for path in ("/health", "/methods", "/models", "/docs", "/redoc", "/openapi.json"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["x-request-id"]
    assert calls == {"encoder": 0, "scorer": 0}
    assert not manager.initialized_names
    assert {m["method"] for m in client.get("/methods").json()["methods"]} == {
        "bm25",
        "dense",
        "hybrid",
        "reranked",
        "router",
    }
    assert client.get("/health").json()["resources"]["dense"]["state"] == "not_loaded"


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid", "reranked", "router"])
def test_all_search_methods(client, setup, method):
    _, manager, _ = setup
    response = client.post("/search", json={"query": "ERR_42", "method": method, "top_k": 3})
    assert response.status_code == 200, response.text
    output = SearchResponse.model_validate(response.json())
    assert output.request_id == response.headers["x-request-id"]
    assert len(output.documents) == 3
    assert [r.rank for r in output.documents] == [1, 2, 3]
    assert all(
        r.title and r.text and r.metadata["dataset"] == "synthetic-smoke" for r in output.documents
    )
    assert output.latency.initialization_ms >= 0 and output.latency.retrieval_ms >= 0
    assert str(manager.settings.dense_index) not in response.text
    if method == "router":
        assert output.selected_strategy == "hybrid"
        assert output.routing and output.routing.explanation
        assert (
            output.routing.policy_record_sha256 == manager.get("policy")[0]["policy_record_sha256"]
        )
        assert not output.routing.reranker_invoked


def test_api_preserves_direct_bm25_and_default_top_k(client, setup):
    direct = BM25Retriever(read_jsonl(SMOKE / "corpus.jsonl", Document).values()).search(
        "ERR_42", 3
    )
    output = client.post("/search", json={"query": "ERR_42"}).json()
    assert output["top_k"] == 3
    assert [(r["document_id"], r["rank"], r["score"]) for r in output["documents"]] == [
        (r.document_id, r.rank, r.score) for r in direct
    ]
    assert output["documents"][0]["document_id"] == "d2"


def test_compare_rankings_no_relevance_metrics(client):
    response = client.post("/compare", json={"query": "ERR_42", "top_k": 3})
    assert response.status_code == 200, response.text
    output = CompareResponse.model_validate(response.json())
    assert len(output.results) == 5 and len(output.agreement) == 10
    by_method = {r.method.value: r for r in output.results}
    assert by_method["router"].documents == by_method["hybrid"].documents
    assert all(0 <= a.jaccard <= 1 and 0 <= a.same_position_fraction <= 1 for a in output.agreement)
    assert not any(
        k.startswith(("recall", "ndcg", "precision", "mrr"))
        for r in response.json()["results"]
        for k in r
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "  "},
        {"query": None},
        {"query": 42},
        {"query": "x" * 1025},
        {"query": "x", "method": "unknown"},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 21},
        {"query": "x", "top_k": True},
        {"query": "x", "top_k": "10"},
        {"query": "x", "model_path": "C:/private/sensitive"},
        {"query": "x", "qrels": {"secret": 1}},
    ],
)
def test_invalid_search(client, payload):
    response = client.post("/search", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert "C:/private/sensitive" not in response.text


@pytest.mark.parametrize("methods", [[], ["bm25", "bm25"], ["unknown"], ["bm25"] * 6])
def test_invalid_compare(client, methods):
    assert client.post("/compare", json={"query": "x", "methods": methods}).status_code == 422


def test_missing_dataset_still_has_health(setup, tmp_path):
    settings, _, _ = setup
    settings = settings.model_copy(update={"dataset_dir": tmp_path / "missing-private-corpus"})
    with TestClient(create_app(settings)) as http:
        assert http.get("/health").json()["status"] == "degraded"
        response = http.post("/search", json={"query": "ERR_42"})
        assert response.status_code == 503
        assert "missing-private-corpus" not in response.text
        assert "Traceback" not in response.text


def test_missing_index_does_not_load_model_and_bm25_survives(client, setup):
    settings, manager, calls = setup
    settings.dense_index.unlink()  # Only a test fixture's own generated cache.
    response = client.post("/search", json={"query": "ERR_42", "method": "dense"})
    assert response.status_code == 503
    assert calls["encoder"] == 0
    assert client.post("/search", json={"query": "ERR_42", "method": "bm25"}).status_code == 200
    assert manager.snapshot()["resources"]["dense"]["state"] == "failed"
    assert "hybrid" not in manager.snapshot()["available_methods"]


def test_missing_model_files_safe_error(setup, tmp_path):
    settings, _, _ = setup
    settings = settings.model_copy(update={"dense_model_dir": tmp_path / "private-missing-model"})
    with TestClient(create_app(settings)) as http:
        response = http.post("/search", json={"query": "test", "method": "dense"})
        assert response.status_code == 503
        assert "private-missing-model" not in response.text
        assert http.get("/models").json()["models"]["dense"]["state"] == "failed"


@pytest.mark.parametrize("failure", [RuntimeError, ImportError])
def test_model_load_failure_cached_not_leaked(setup, failure):
    settings, _, _ = setup
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise failure("C:/private/model SECRET_QUERY")

    manager = ModelService(settings, broken, lambda *a, **k: Scorer())
    with TestClient(create_app(settings, lambda _: manager)) as http:
        for _ in range(2):
            response = http.post("/search", json={"query": "not logged", "method": "dense"})
            assert response.status_code == 503
            assert "C:/private" not in response.text and "SECRET_QUERY" not in response.text
        assert len(calls) == 1
        assert http.get("/health").json()["status"] == "degraded"


def test_router_does_not_initialize_reranker(client, setup):
    response = client.post("/search", json={"query": "ERR_42", "method": "router"})
    assert response.status_code == 200
    assert setup[2] == {"encoder": 1, "scorer": 0}
    assert client.get("/models").json()["models"]["reranker"]["state"] == "not_loaded"


def test_model_cache_reused_across_search_and_compare(client, setup):
    for _ in range(2):
        assert client.post("/compare", json={"query": "ERR_42", "top_k": 3}).status_code == 200
    response = client.post("/search", json={"query": "ERR_42", "method": "dense"})
    assert response.json()["latency"]["initialization_ms"] == 0
    assert response.json()["initialized_resources"] == []
    assert setup[2] == {"encoder": 1, "scorer": 1}


def test_concurrent_initialization_once(setup):
    settings, _, _ = setup
    calls = []
    start = threading.Barrier(4)

    def factory(*args, **kwargs):
        calls.append(1)
        sleep(0.03)
        return Encoder()

    manager = ModelService(settings, factory, lambda *a, **k: Scorer())

    def initialize():
        start.wait(timeout=3)
        return manager.get("dense")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: initialize(), range(4)))
    assert len(calls) == 1
    assert all(result is results[0] for result in results)


@pytest.mark.parametrize("queued", [0, 1])
def test_bounded_capacity_queue_and_release(setup, queued):
    settings, _, _ = setup
    manager = ModelService(
        settings.model_copy(update={"max_queued_requests": queued, "max_queue_wait_seconds": 0})
    )
    search = SearchService(manager)
    with manager.execution():
        assert manager.snapshot()["status"] == "ok"
        with pytest.raises(ServiceError) as exc:
            search.search("ERR_42", "bm25")
        assert exc.value.status_code == 429
    assert search.search("ERR_42", "bm25")["documents"]


def test_body_limits_malformed_json_unknown_routes(client):
    assert client.post("/search", content=b"x" * 17000).status_code == 413
    assert client.post("/search", content=iter([b"x" * 9000, b"x" * 9000])).status_code == 413
    assert client.post("/search", content=b"{broken").status_code == 422
    assert client.get("/missing").status_code == 404
    assert client.get("/search").status_code == 405


def test_chunked_limit_before_downstream():
    messages = iter(
        [
            {"type": "http.request", "body": b"a" * 700, "more_body": True},
            {"type": "http.request", "body": b"b" * 700, "more_body": False},
        ]
    )
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    async def forbidden(*args):
        raise AssertionError("Oversized body must not reach application")

    scope = {"type": "http", "method": "POST", "headers": [], "state": {}}
    asyncio.run(BoundaryMiddleware(forbidden, 1024)(scope, receive, send))
    assert sent[0]["status"] == 413
    assert len(json.loads(sent[1]["body"])["request_id"]) == 32


def test_unexpected_error_has_safe_consistent_response(setup):
    settings, manager, _ = setup

    def broken():
        raise RuntimeError("PRIVATE_PATH SECRET_BODY")

    manager.models = broken
    with TestClient(create_app(settings, lambda _: manager)) as http:
        response = http.get("/models")
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
        assert response.json()["request_id"] == response.headers["x-request-id"]
        assert "PRIVATE_PATH" not in response.text and "SECRET_BODY" not in response.text


def test_configuration_and_cors(setup, monkeypatch):
    for values in ({"default_top_k": 21}, {"cors_origins": ("*",)}, {"rerank_candidates": 51}):
        with pytest.raises(ValidationError):
            Settings(**values)
    monkeypatch.setenv("RANKFORGE_PORT", "8123")
    assert Settings.from_env().port == 8123
    settings, manager, _ = setup
    settings = settings.model_copy(update={"cors_origins": ("http://localhost:3000",)})
    with TestClient(create_app(settings, lambda _: manager)) as http:
        allowed = http.get("/health", headers={"Origin": "http://localhost:3000"})
        denied = http.get("/health", headers={"Origin": "https://untrusted.example"})
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "access-control-allow-origin" not in denied.headers


def test_services_comparison_and_shutdown(setup):
    _, manager, _ = setup
    search = SearchService(manager)
    result = ComparisonService(search).compare("ERR_42", ["bm25", "dense"], 3)
    assert len(result["agreement"]) == 1
    manager.close()
    with pytest.raises(ServiceError):
        search.search("ERR_42", "bm25")


def test_logging_fields_are_structured():
    record = logging.LogRecord("rankforge.service", logging.INFO, "", 0, "http_request", (), None)
    record.fields = {"method": "POST", "status_code": 200, "duration_ms": 1.2}
    body = json.loads(JsonFormatter().format(record))
    assert body["event"] == "http_request" and body["status_code"] == 200
    assert "query" not in body and "body" not in body


def test_health_remains_responsive_during_initialization(setup):
    settings, _, _ = setup
    entered, release = threading.Event(), threading.Event()

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return Encoder()

    manager = ModelService(settings, slow)
    with ThreadPoolExecutor(max_workers=2) as executor:
        loading = executor.submit(manager.get, "dense")
        try:
            assert entered.wait(timeout=3)
            snapshot = executor.submit(manager.snapshot).result(timeout=1)
            assert snapshot["resources"]["encoder"]["state"] == "initializing"
        finally:
            release.set()
        assert loading.result(timeout=3)


def test_http_busy_is_safe_429(setup):
    settings, _, _ = setup
    settings = settings.model_copy(update={"max_queued_requests": 0})
    manager = ModelService(settings)
    with TestClient(create_app(settings, lambda _: manager)) as http:
        with manager.execution():
            response = http.post("/search", json={"query": "ERR_42"})
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "server_busy"
        assert response.headers["x-request-id"] == response.json()["request_id"]


def test_default_cors_disabled_and_shutdown_releases(client, setup):
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" not in response.headers
    manager = setup[1]
    manager.get("bm25")
    manager.close()
    assert manager.snapshot()["resources"]["bm25"]["state"] == "released"


def test_recorded_openapi_examples_match_response_schemas(client):
    from app.api.schemas import ErrorResponse

    schema = client.get("/openapi.json").json()
    for route, model in (("/search", SearchResponse), ("/compare", CompareResponse)):
        example = schema["paths"][route]["post"]["responses"]["200"]["content"]["application/json"][
            "example"
        ]
        model.model_validate(example)
        assert example.get("documents", example.get("results"))
    ErrorResponse.model_validate(
        schema["paths"]["/search"]["post"]["responses"]["422"]["content"]["application/json"][
            "example"
        ]
    )


@pytest.mark.parametrize("header,status", [(b"-1", 400), (b"invalid", 400), (b"9" * 5000, 413)])
def test_invalid_or_huge_content_length_is_bounded(header, status):
    sent = []

    async def receive():
        raise AssertionError("Rejected before reading request body")

    async def send(message):
        sent.append(message)

    async def forbidden(*args):
        raise AssertionError("Rejected before application invocation")

    scope = {"type": "http", "method": "POST", "headers": [(b"content-length", header)]}
    asyncio.run(BoundaryMiddleware(forbidden, 1024)(scope, receive, send))
    assert sent[0]["status"] == status
