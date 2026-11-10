# Phase 5 local HTTP service

Phase 7A adds [local Docker packaging](docker_deployment.md). Optional `RANKFORGE_DEFAULT_METHOD` (bm25 by default) selects the method when omitted from `/search`; explicit methods and null rejection are preserved. `RANKFORGE_LOG_LEVEL` defaults to info. Native startup/lazy resource behavior is unchanged; the container adds required-file preflight and uses separate Linux disk indexes.

RankForge exposes existing retrievers through FastAPI. This is a locally validated CPU research service. The experiment CLIs remain independent; HTTP never loads qrels, calculates relevance metrics, downloads models, builds dense indexes or writes experiment results.

## Prepare and start (Windows PowerShell)

Run from the repository root. Reuse the existing environment and prepared resources; these commands do not recreate them.

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[api,neural,dev]' --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip check
# Prepare only resources that are absent; existing commands validate/reuse them.
.\.venv\Scripts\python.exe -m app.data.download
.\.venv\Scripts\python.exe -m app.data.models --dense-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --rerank-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
.\.venv\Scripts\python.exe -m app.retrieval.dense --dataset data/public/scifact --cache indexes/scifact-minilm.npz --batch-size 16 --threads 4
.\.venv\Scripts\python.exe -m app.api
```

The dataset download fallback using `Invoke-WebRequest` and the archive importer is in the README. Dataset and model attribution/terms remain documented in the data/Phase 2 methodology. No download was needed for Phase 5 verification. Keep the existing `experiments/configs/phase4-transfer` capsule; do not refit a policy for the API. If a new checkout lacks the capsule, restore the preserved capsule and source policy first, or use BM25 without router support. `app.evaluation.transfer_policy` is the existing explicit offline freeze workflow, not an API operation.

The documented startup command binds `127.0.0.1:8000`, uses one worker, disables Uvicorn access logs and honors `RANKFORGE_HOST`/`RANKFORGE_PORT`. Importing `app.api.main` creates configuration/routes only; lifespan creates one empty resource manager. Press Ctrl+C once to let Uvicorn drain requests and run lifespan cleanup. Model references are released on shutdown; no cache or experiment files are removed. A forced process termination cannot run cleanup.

Equivalent explicit Uvicorn invocation (supply host/port on its command line):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Open `http://127.0.0.1:8000/docs` or `/redoc`; the machine-readable schema is `/openapi.json`. Swagger/ReDoc HTML was verified over HTTP. Their default JavaScript/CSS use public CDNs, so interactive browser rendering needs network access; assets are not bundled offline.

## Call every endpoint

In a second PowerShell terminal:

```powershell
$base = 'http://127.0.0.1:8000'
Invoke-RestMethod "$base/health" | ConvertTo-Json -Depth 8
Invoke-RestMethod "$base/methods" | ConvertTo-Json -Depth 8
Invoke-RestMethod "$base/models" | ConvertTo-Json -Depth 8
$search = @{ query = 'What evidence links vitamin D deficiency to disease?'; method = 'hybrid'; top_k = 10 } | ConvertTo-Json
Invoke-RestMethod "$base/search" -Method Post -ContentType 'application/json' -Body $search | ConvertTo-Json -Depth 10
$routed = @{ query = 'What evidence links vitamin D deficiency to disease?'; method = 'router'; top_k = 10 } | ConvertTo-Json
Invoke-RestMethod "$base/search" -Method Post -ContentType 'application/json' -Body $routed | ConvertTo-Json -Depth 10
$compare = @{ query = 'What evidence links vitamin D deficiency to disease?'; methods = @('bm25','dense','hybrid','reranked','router'); top_k = 10 } | ConvertTo-Json
Invoke-RestMethod "$base/compare" -Method Post -ContentType 'application/json' -Body $compare | ConvertTo-Json -Depth 12
```

`/search` returns `documents` with `document_id`, rank, score, title, original text and public dataset label; `selected_strategy`, score type, algorithm explanation, optional routing explanation/features/policy identity, initialized resource names and latency fields accompany them. Recorded authored smoke responses are included as OpenAPI examples and validated by tests. Their incidental timings are examples, not public-data benchmark measurements.

Scores are BM25 Okapi, normalized cosine, RRF or cross-encoder raw logits. They are not probabilities and are not directly comparable across methods. Ranking explanations describe the algorithm and frozen decision rule, not causal document explanations or term attributions. Document metadata contains the configured public dataset alias; internal paths and qrels are absent.

`/compare` runs each requested complete pipeline independently in request order, against the same corpus. It returns one result per method and all method pairs: shared IDs, set Jaccard (`intersection/union`) and matching-position fraction (`equal IDs at the same rank / larger returned list size`). Empty/empty agreement is defined as zero. A single requested method has no pairs. No query-result or embedding reuse is introduced between methods. Any requested-method error fails the comparison as a whole; previously initialized healthy resources remain cached. This is ranking agreement, not Recall/nDCG or a scientific timing comparison.

## Lifecycle and configuration

`app/core/config.py` is the centralized frozen configuration. Values come from process environment, never HTTP bodies or query parameters. `.env.example` lists defaults but is not automatically loaded; set `$env:RANKFORGE_...` in PowerShell. Restart to change settings.

| Variable suffix after `RANKFORGE_` | Default | Purpose |
| --- | --- | --- |
| DATASET_DIR / DATASET_LABEL | data/public/scifact / scifact | Corpus directory and safe public label |
| DENSE_MODEL_DIR / RERANK_MODEL_DIR | models/dense / models/reranker | Prepared local model snapshots |
| DENSE_INDEX | indexes/scifact-minilm.npz | Prepared compatible read-only vector cache |
| POLICY_DIR | experiments/configs/phase4-transfer | Existing validated frozen capsule |
| DEFAULT_TOP_K | 10 | Default result count, 1..20 |
| BATCH_SIZE / THREADS | 16 / 4 | Existing neural adapter settings |
| CANDIDATE_DEPTH / RERANK_CANDIDATES | 50 / 20 | Hybrid source depth / bounded cross-encoder pool |
| MAX_BODY_BYTES | 16384 | Actual buffered HTTP body limit, 1024..65536 |
| MAX_QUEUED_REQUESTS | 2 | Additional admitted searches beyond the active one, 0..16 |
| MAX_QUEUE_WAIT_SECONDS | 2 | Queue acquisition timeout, 0..60 |
| HOST / PORT | 127.0.0.1 / 8000 | `python -m app.api` bind settings |
| CORS_ORIGINS | empty | Comma-separated exact HTTP(S) origins; no wildcard/credentials |

Example:

```powershell
$env:RANKFORGE_PORT = '8123'
$env:RANKFORGE_DEFAULT_TOP_K = '5'
$env:RANKFORGE_CORS_ORIGINS = 'http://localhost:3000'
.\.venv\Scripts\python.exe -m app.api
```

Queries must contain non-whitespace text and at most 1,024 characters. Input is preserved exactly, not rewritten. `top_k` is a strict JSON integer from 1 to 20; booleans/string integers are rejected. `/compare` accepts 1..5 unique supported methods. Extra body fields, including paths/model names/qrels, are rejected. A requested reranked `top_k` cannot exceed its configured pool. The existing reranker requires its pool to be smaller than the corpus and no larger than hybrid source depth. These defaults suit SciFact/NFCorpus, not the six-document synthetic fixture.

For a BM25-only synthetic session, use a **fresh PowerShell terminal** and the following settings. This requires `.[api,dev]` only; no neural download is necessary. Neural requests without the separate compatible smoke index fail safely.

```powershell
$env:RANKFORGE_DATASET_DIR = 'data/smoke'
$env:RANKFORGE_DATASET_LABEL = 'synthetic-smoke'
$env:RANKFORGE_DEFAULT_TOP_K = '3'
$env:RANKFORGE_CANDIDATE_DEPTH = '5'
$env:RANKFORGE_RERANK_CANDIDATES = '3'
$env:RANKFORGE_DENSE_INDEX = 'indexes/smoke-minilm.npz'
.\.venv\Scripts\python.exe -m app.api
# In the second terminal, POST /search with query ERR_42, method bm25, top_k 3.
```

Resources initialize lazily once per process under a reentrant initialization lock, and remain cached. Dense cache identity/checksums, original models and canonical policy/implementation guards are reused unchanged. The API requires an existing dense index and never takes the offline index-building path. Model adapters use local files only, with remote custom code disabled. Failed initialization is sticky until restart; status gives a safe resource error code. The hybrid-only frozen router loads no cross-encoder unless another requested method needs it. A first successful search is an explicit warmup for that pipeline; there is no automatic expensive startup warmup.

Searches are serialized per worker, including `/compare`, so shared neural models/indexes cannot be mutated or invoked concurrently through these services. Bounded admission and queue timeout return 429 under contention. Health/model status uses a separate short-held lock and does not wait for model initialization or hash model weights. Status checks inspect file/dependency presence, not full validity: `available_methods` is provisional until first use validates resources. `/health` is liveness/status, not a readiness guarantee for all five methods. It returns 200 with `degraded` for a missing corpus or known initialization failure; missing optional neural resources can leave a healthy BM25 service.

## Timing, errors and logs

All measurements use monotonic clocks. `latency.initialization_ms` includes dependency setup once, without nested double counting. `retrieval_ms` is the complete selected pipeline excluding that lazy initialization; `rerank_ms` is a component of it, not an additional duration. `serialization_ms` constructs the document payload; it excludes FastAPI response validation/JSON encoding. `queue_ms` is lock/admission waiting. `total_service_ms` includes service orchestration and waiting, excluding HTTP JSON encoding and network transit. For comparison, the queue is reported once outside individual method results. Resource status durations include their dependencies and should not be summed.

`X-Process-Time-Ms` covers the middleware request through response headers, including FastAPI response encoding. Verification's `http_round_trip_ms` measures client POST through complete response receipt, excluding later client JSON decoding. It is not equivalent to offline benchmark latency. First-use costs follow shared-resource order: a first standalone dense call can be warm if router already initialized dense resources.

Errors share `{ "request_id": "...", "error": { "code": "...", "message": "...", "details": [] } }`. Validation gives 422; invalid Content-Length 400; body limit 413; unsupported route/verb 404/405; capacity/queue timeout 429; missing/incompatible resource or inference failure 503; unexpected errors 500. Paths, raw exception messages, stacks, input bodies and environment values are omitted. Validation details include only field/type/message. Every request gets a server-generated ID in the body and `X-Request-ID`; JSON logs carry that ID, route template, method, status and duration. Logs omit complete query text and request bodies by default. Uvicorn access logging is disabled by documented commands.

Common recovery: install `.[api]` for missing FastAPI; `.[neural]` plus prepared snapshots for neural dependency/model errors; prepare the dataset for corpus errors; use the offline dense CLI with the matching corpus/model/cache for missing/stale indexes; restore the frozen capsule/source implementation for policy errors; stop the other process or change port for a bind error. Restart after repairing failed resources. Do not bypass identity checks or certificate verification. If the configured package index cannot resolve, the explicit public PyPI index above worked here; the verified offline editable reinstall used `--no-build-isolation --no-index` only after all dependencies and build tools were installed.

## Verify and scope

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/verify_api.py
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --replay experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104
.\.venv\Scripts\python.exe -m app.evaluation.phase3 --replay experiments/results/phase3/phase3-20261031T065803Z-72804b2af31a
.\.venv\Scripts\python.exe -m app.evaluation.phase4 --replay experiments/results/phase4/phase4-20261031T071828Z-46027d7d4330
```

Unit/integration tests use authored corpus files, prepared tiny vectors and model doubles; no public download or large-model load occurs. The explicit verification script starts the actual `python -m app.api` subprocess on an ephemeral loopback port, forces model libraries offline, exercises every endpoint/docs/invalid requests, validates response schemas, compares all five methods to saved SciFact query 1241 ranks/scores, checks frozen identity and model initialization events, and checks old artifact/model/index bytes. It saves a unique Phase 5 run, real responses/OpenAPI/logs/hash manifests and a unique portable report. It terminates only its own child on Windows; lifespan cleanup is separately tested with TestClient. `--warm-requests 5` changes sample count; this is not an evaluation sweep or load test.

Implemented files: `app/api/{main,routes,schemas,examples,dependencies,errors,middleware,__main__}.py`; `app/core/{config,errors,logging}.py`; `app/services/{model_service,search_service,comparison_service}.py`, package initializers, `tests/test_api.py`, `scripts/verify_api.py`, `.env.example` and optional API dependencies in `pyproject.toml`. README, architecture, methodology, decision/result docs and a separate environment freeze document the milestone. Retrieval algorithms, metrics, model adapters, policy selection and old evaluation CLIs are unchanged.

Measured results and exact report links are in [experiment results](experiment_results.md). Limits: one local CPU worker, serialized searches, shared first-use costs, no hard inference cancellation or upload deadline, no throughput/peak-memory/load test, no browser/CDN rendering test, no fresh-environment installation test. No authentication or internet deployment was added; bind to loopback. Recommended next milestone is repeatable mixed-query concurrency/timeout and resource-budget validation before a dashboard or deployment. Broader-domain retrieval evaluation remains separate analytical work.


## Phase 6 follow-up

[Local reliability validation](api_reliability.md) adds mixed-query concurrency/resource measurements and deterministic fault probes. Resource status now includes initialization attempt/success counters; health includes active/queued/stopping activity. Shutdown signals stopping before waiting for active work, rejects new/queued searches with 503, and releases resources only after active retrieval completes. No inference behavior or frozen policy changed. Client disconnects do not imply that CPU work stopped; see the Phase 6 guide for verified recovery behavior and measured limits.
