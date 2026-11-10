"""Isolated authored-BM25 probe of exception/request references; no forced GC."""

import gc
import json
import socket
import sys
import threading
import types
from pathlib import Path

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api.main import create_app  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402


def census():
    objects = gc.get_objects()
    targets = [
        o
        for o in objects
        if type(o).__module__ == "starlette.requests" and type(o).__name__ == "Request"
    ]
    details = []
    for obj in targets[:8]:
        refs = gc.get_referrers(obj)
        descriptions = []
        for ref in refs:
            if ref is objects or ref is targets:
                continue
            if isinstance(ref, types.FrameType):
                descriptions.append(
                    {
                        "type": "frame",
                        "function": ref.f_code.co_name,
                        "file": ref.f_code.co_filename,
                        "line": ref.f_lineno,
                    }
                )
            elif isinstance(ref, dict):
                descriptions.append(
                    {
                        "type": "dict",
                        "keys_pointing_to_request": [str(k) for k, v in ref.items() if v is obj][
                            :10
                        ],
                    }
                )
            else:
                descriptions.append({"type": type(ref).__name__})
        details.append(descriptions)
    return {
        "request_count": len(targets),
        "gc_generation_counts": list(gc.get_count()),
        "referrer_samples": details,
        "notes": "The census creates temporary observing references; no object payloads or query texts are recorded.",
    }


def main():
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    manager = ModelService(
        Settings(
            dataset_dir=ROOT / "data/smoke",
            dataset_label="authored-probe",
            max_queued_requests=0,
            default_top_k=3,
            candidate_depth=5,
            rerank_candidates=3,
        )
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(manager.settings, lambda _: manager),
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    while not server.started:
        threading.Event().wait(0.01)
    report = {
        "scope": "Isolated real HTTP, authored six-document BM25 corpus. Qualitative reference diagnosis, not real-model RSS comparison. No forced GC or artificial allocation pressure."
    }
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
            assert (
                client.post(
                    "/search", json={"query": "ERR_42", "method": "bm25", "top_k": 3}
                ).status_code
                == 200
            )
            report["baseline"] = census()
            with manager.execution():
                statuses = [
                    client.post(
                        "/search", json={"query": "ERR_42", "method": "bm25", "top_k": 3}
                    ).status_code
                    for _ in range(50)
                ]
            assert set(statuses) == {429}
            report["after_50_rejections"] = census()
            threading.Event().wait(5)
            report["after_5s_idle"] = census()
            successes = [
                client.post(
                    "/search", json={"query": "ERR_42", "method": "bm25", "top_k": 3}
                ).status_code
                for _ in range(100)
            ]
            assert set(successes) == {200}
            report["after_100_normal_requests"] = census()
            report["statuses"] = {"429": 50, "200": 101}
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        report["graceful_shutdown"] = not thread.is_alive()
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: v["request_count"]
                for k, v in report.items()
                if isinstance(v, dict) and "request_count" in v
            }
        )
    )


if __name__ == "__main__":
    main()
