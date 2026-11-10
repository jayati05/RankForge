"""Offline regression and real-loopback fault probes with authored model doubles."""

import asyncio
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import uvicorn
import test_api as api_fixtures


from app.api.main import create_app
from app.core.errors import ServiceError
from app.evaluation.api_load import compare, post_control, serving_pid
from app.evaluation.api_workload import METHODS, aggregate, fingerprint, queries, schedule, weights
from app.services.model_service import ModelService
from app.services.search_service import SearchService


setup = api_fixtures.setup
isolated_log_handlers = api_fixtures.isolated_log_handlers


def test_shutdown_rejects_new_and_queued_work_while_active_finishes(setup):
    _, manager, _ = setup
    entered, release = threading.Event(), threading.Event()

    def active():
        with manager.execution():
            entered.set()
            assert release.wait(timeout=5)
            return manager.get("bm25").search("ERR_42", 3)

    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(active)
        assert entered.wait(timeout=2)
        queued = executor.submit(SearchService(manager).search, "ERR_42", "bm25", 3)
        deadline = time.monotonic() + 2
        while manager.snapshot()["activity"]["queued"] != 1:
            assert time.monotonic() < deadline
            threading.Event().wait(0.001)
        closing = executor.submit(manager.close)
        try:
            assert manager._stopping.wait(timeout=2)
            assert not closing.done()
            assert manager.snapshot()["status"] == "stopping"
            with pytest.raises(ServiceError) as error:
                SearchService(manager).search("ERR_42", "bm25", 3)
            assert error.value.status_code == 503
        finally:
            release.set()
        assert first.result(timeout=3)
        with pytest.raises(ServiceError) as error:
            queued.result(timeout=3)
        assert error.value.code == "service_stopping"
        closing.result(timeout=3)
    snapshot = manager.snapshot()
    assert snapshot["activity"] == {"active": 0, "queued": 0, "stopping": True}
    assert snapshot["status"] == "stopped"


def test_failed_attempt_counter_is_sticky_and_dependency_message_safe(setup):
    settings, _, _ = setup

    def fail(*args, **kwargs):
        raise ImportError("C:/private/secret-model")

    manager = ModelService(settings, fail)
    for _ in range(2):
        with pytest.raises(ServiceError) as error:
            manager.get("dense")
        assert "secret-model" not in error.value.message
    snapshot = manager.snapshot()
    assert snapshot["resources"]["encoder"]["initialization_attempts"] == 1
    assert snapshot["resources"]["encoder"]["initialization_count"] == 0
    assert snapshot["resources"]["dense"]["initialization_attempts"] == 1


@pytest.fixture
def live(setup):
    settings, manager, _ = setup
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(settings, lambda _: manager),
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline
        threading.Event().wait(0.01)
    try:
        yield f"http://127.0.0.1:{port}", manager, server, thread
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive()


@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_live_disconnect_keeps_slow_inference_serialized_and_recovers(live, monkeypatch, mode):
    url, manager, _, _ = live
    retriever = manager.get("reranked")
    original = retriever.scorer.score
    expected = retriever.search("ERR_42", 3)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow(pairs, batch_size):
        entered.set()
        assert release.wait(timeout=5)
        try:
            return original(pairs, batch_size)
        finally:
            finished.set()

    monkeypatch.setattr(retriever.scorer, "score", slow)

    async def probe():
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
            task = asyncio.create_task(
                client.post(
                    "/search",
                    json={"query": "ERR_42", "method": "reranked", "top_k": 3},
                    timeout=0.2 if mode == "timeout" else 5,
                )
            )
            assert await asyncio.to_thread(entered.wait, 2)
            try:
                if mode == "cancel":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    with pytest.raises(httpx.ReadTimeout):
                        await task
                assert not finished.is_set()
                health = (await client.get("/health")).json()
                assert health["activity"]["active"] == 1
                assert health["activity"]["queued"] == 0
                assert health["resources"]["scorer"]["initialization_count"] == 1
            finally:
                release.set()
            assert await asyncio.to_thread(finished.wait, 2)
            monkeypatch.setattr(retriever.scorer, "score", original)
            response = await client.post(
                "/search", json={"query": "ERR_42", "method": "reranked", "top_k": 3}
            )
            assert response.status_code == 200
            assert [(r["document_id"], r["score"]) for r in response.json()["documents"]] == [
                (r.document_id, r.score) for r in expected
            ]

    asyncio.run(probe())
    assert manager.snapshot()["activity"]["active"] == 0


def test_live_inference_exception_releases_capacity_and_recovers(live, monkeypatch):
    url, manager, _, _ = live
    retriever = manager.get("reranked")
    original = retriever.scorer.score

    def broken(*args, **kwargs):
        raise RuntimeError("PRIVATE_QUERY C:/private/weights Traceback")

    monkeypatch.setattr(retriever.scorer, "score", broken)
    with httpx.Client(base_url=url, trust_env=False) as client:
        response = client.post(
            "/search", json={"query": "ERR_42", "method": "reranked", "top_k": 3}
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "inference_failed"
        assert not any(
            word in response.text for word in ("PRIVATE_QUERY", "C:/private", "Traceback")
        )
        assert response.json()["request_id"] == response.headers["x-request-id"]
        monkeypatch.setattr(retriever.scorer, "score", original)
        assert (
            client.post(
                "/search", json={"query": "ERR_42", "method": "reranked", "top_k": 3}
            ).status_code
            == 200
        )
    assert manager.snapshot()["resources"]["scorer"]["initialization_count"] == 1
    assert manager.snapshot()["activity"]["active"] == 0


def test_live_graceful_shutdown_drains_active_inference(live, monkeypatch):
    url, manager, server, server_thread = live
    retriever = manager.get("reranked")
    entered, release = threading.Event(), threading.Event()
    original = retriever.scorer.score

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return original(*args, **kwargs)

    monkeypatch.setattr(retriever.scorer, "score", slow)
    with ThreadPoolExecutor(max_workers=1) as executor:

        def request():
            with httpx.Client(trust_env=False, timeout=5) as client:
                return client.post(
                    url + "/search", json={"query": "ERR_42", "method": "reranked", "top_k": 3}
                )

        future = executor.submit(request)
        try:
            assert entered.wait(timeout=2)
            server.should_exit = True
            assert server_thread.is_alive()
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 200
    server_thread.join(timeout=5)
    assert not server_thread.is_alive()
    assert manager.snapshot()["status"] == "stopped"
    assert manager.snapshot()["resources"]["scorer"]["state"] == "released"


def test_schedule_reproducible_balanced_repeated_and_label_free(tmp_path):
    public = [{"_id": qid, "text": "Claim " + qid} for qid in ("94", "1241", "1088", "1024")]
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in public))
    pool = queries(tmp_path)  # No qrels/corpus file exists in this fixture.
    distribution = weights(",".join(m + ":1" for m in METHODS))
    a = schedule(pool, 100, distribution, 42)
    assert fingerprint(a) == fingerprint(schedule(pool, 100, distribution, 42))
    assert fingerprint(a) != fingerprint(schedule(pool, 100, distribution, 43))
    assert all(sum(r["method"] == method for r in a) == 20 for method in METHODS)
    assert len({r["query"]["id"] for r in a}) == 8
    assert len({r["query"]["text"] for r in a}) < len(a)
    assert {q["category"] for q in pool} >= {
        "punctuation",
        "numbers-technical",
        "short-keywords",
        "natural-language",
    }


def test_aggregation_counts_rejections_and_timeouts_without_success_bias():
    rows = [
        {"outcome": "success", "status": 200, "latency_ms": 10},
        {"outcome": "http_error", "status": 429, "latency_ms": 1},
        {"outcome": "timeout", "status": None, "latency_ms": 100},
    ]
    stats = aggregate(rows, 2)
    assert stats["successful"] == 1 and stats["failed"] == 2 and stats["timed_out"] == 1
    assert stats["statuses"] == {"200": 1, "429": 1, "timeout": 1}
    assert stats["attempts_per_second"] == 1.5 and stats["successful_per_second"] == 0.5
    assert stats["success_latency"]["p99_ms"] == 10
    assert stats["all_latency"]["p99_ms"] > 90
    assert aggregate([], 1)["success_latency"]["mean_ms"] is None


@pytest.mark.parametrize("distribution", ["dense:0", "unknown:1", "bm25:1,bm25:2"])
def test_invalid_method_weights(distribution):
    with pytest.raises(ValueError):
        weights(distribution)


def test_correctness_requires_same_ids_order_and_scores():
    hits = [{"document_id": "001", "rank": 1, "score": 0.2}]
    assert compare(hits, [{**hits[0], "score": 0.200001}])
    assert not compare(hits, [{**hits[0], "score": 0.21}])
    assert not compare(hits, [{**hits[0], "document_id": "1"}])
    assert not compare(hits, [])


def test_control_file_commands_are_atomic_and_sequenced(tmp_path):
    path = tmp_path / "control.json"
    post_control(path, 0, "idle")
    post_control(path, 1, "initialize")
    assert json.loads(path.read_text()) == {"sequence": 1, "command": "initialize"}
    assert not path.with_suffix(".tmp").exists()
    post_control(path, 2, "stop")
    assert json.loads(path.read_text())["command"] == "stop"


def test_memory_selects_actual_serving_pid_instead_of_windows_launcher():
    assert serving_pid("launcher PID 100\nINFO:     Started server process [200]\n") == 200
    with pytest.raises(RuntimeError, match="refusing launcher"):
        serving_pid("launcher PID 100, no verified serving process")
