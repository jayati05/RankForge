# Phase 6: local API reliability validation

The Phase 6 harness exercises the existing FastAPI/service/retriever architecture over real loopback HTTP, without qrels or relevance-based request scheduling. This is a single-machine research validation, not a production capacity certification. Exact measured outcomes are appended in `experiment_results.md`; the decision log preserves intermediate failures too.

## Reproduce on Windows

Reuse the Phase 5 virtual environment, local SciFact corpus, prepared vector cache, model snapshots and frozen transfer capsule. No additional package or model download is required beyond the existing `api`, `neural`, `analysis` and `dev` extras. See `api_service.md` for explicit preparation commands. The harness starts and stops its own child on an ephemeral loopback port; no separately running API is needed.

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.api_load --requests 100 --clients 1,2,4,8,16 --timeout 30 --initialization-timeout 900 --warmup 10 --seed 42
```

For a smaller initial run or different distribution:

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.api_load --requests 20 --clients 1,2 --timeout 30 --initialization-timeout 900 --warmup 5 --seed 42 --methods 'bm25:2,dense:1,hybrid:2,reranked:1,router:2'
```

`--requests` is the measured count **per concurrency level**, not the total over all levels. Each level reuses the same exact seeded request schedule. Weights are positive integers; requests use a weighted cycle truncated to count, then a seeded shuffle. Query order is independently shuffled with the same local RNG and cycled, creating reproducible repetitions. Client arrival/order and service scheduling remain nondeterministic. `--timeout` is HTTPX's per-I/O-phase timeout, not a hard server computation deadline; `--initialization-timeout` separately governs cold first-use/draining and controlled resource setup. Warmup/correctness/probes are outside measured workload throughput. No retries/backoff conceal errors.

Eight query texts are fixed before requests: four exposed public SciFact compatibility IDs (`94`, `1241`, `1088`, `1024`) from prior comparisons, plus four authored queries covering short keywords, longer questions, numbers/technical terms and punctuation. The public cases previously showed different method rankings. No qrels are opened, no relevance labels are invented, and the authored queries have no quality labels. `workload.json` retains original texts/IDs, category/source and every method assignment; its canonical SHA256 identifies the schedule. This workload is deliberately small and partly biomedical, not a representative sample of all user traffic.

## Method and lifecycle

The direct-correctness child uses existing retrievers, the complete SciFact corpus, unchanged model identities, read-only cache, k=10 and frozen policy. It exits before the measured API child starts, so its models are not included in server memory measurements. A separate server process runs the same `create_app()` factory and Uvicorn. Benchmark-only private atomic control-file commands initialize resources without an inference endpoint and request graceful Uvicorn shutdown; these are not HTTP administration features. The controller polls a small file in its own unique run directory and never blocks on stdin; pipe-reading controller prototypes interfered with native imports on this Windows runtime and were discarded with failed artifacts preserved. Both children force Hugging Face/Transformers offline and retain existing CPU settings (four torch threads, batch 16, source depth 50, RRF 60, cross-encoder pool 20).

Cold health must report zero model/index initializations. Four simultaneous first-use dense requests test single initialization and record queue/capacity rejections. A subsequent router request must use the exact frozen policy while leaving the cross-encoder unloaded. The controlled initialization command then loads remaining resources once. After the configured warmup, all eight queries and five methods are compared to direct results, then measured closed-loop workloads run at the requested levels. Every successful measured response is checked too, followed by all 40 query/method pairs again after load. Exact ordered IDs/ranks and absolute score difference <=1e-5 (relative tolerance zero) are required. Router policy identity is checked; no routing policy is fitted or modified.

Each concurrency level creates that many HTTPX workers sharing a client with enough connections. A worker sends its next assigned request only after the prior attempt ends. No arrival-rate control or fairness claim is made. Latency runs from POST to complete response receipt and excludes later client-side response validation. The experiment waits for active and queued work to drain before moving levels. Throughput is attempts or successful responses divided by measured client batch duration; resource/correctness/warmup/probe work is excluded. Method-specific rates share that batch denominator.

All-attempt and successful-request latency are reported separately. A fast 429 is a failed request, not evidence of faster retrieval. Means and NumPy linear p50/p95/p99 are descriptive; small successful subsets make tails unstable. Status distributions, every non-success, timeouts and transport/validation failures remain visible. An HTTP 200 in server logs after a disconnected client is not counted as client success. The configured serialized engine admits at most one active search and two waiting searches, with a two-second queue timeout; 16 clients do not imply 16 simultaneous inference executions.

## Memory and counts

The native Windows process monitor calls `GetProcessMemoryInfo` for the verified Uvicorn serving PID only (not Windows venv `Popen.pid`, which can be a launcher) (Linux uses `/proc`). Working set/RSS, cumulative OS lifetime peak and private commit are separate values. Samples occur every 100ms and at stage markers: before model loading, after resource initialization, after warmup, during each load level, and after requests finish. Sampled peaks can miss short transients; the OS peak is cumulative rather than attributable to a particular batch. The after-initialization marker includes the earlier cold dense/router first-use searches but precedes the configured warmup. Parent/direct-worker/client/OS memory is excluded, so this is not total machine memory or allocated tensor memory. No long-term leak claim follows from a single pass.

`/health` and `/models` expose per-resource initialization attempts/success counts without loading resources. Logs independently count each successful initialization. Model and index counts must remain one across successful reuse; failed initialization has attempts=1, count=0 and remains sticky until restart. Health also reports `activity.active`, `queued`, `stopping`, using only the short status lock. Existing local artifact/model/index files are hashed before/after; earlier retrieval/evaluation/data implementations are protected by the audit manifest and original replays.

## Cancellation, exceptions and shutdown

The real-model run applies a short client read timeout to a reranked query, cancels an HTTPX task only after health observes active inference, drains the server, then checks unchanged recovery rankings. These verify client-side timeout/disconnect/cancellation and safe subsequent reuse. They do **not** implement an API job-cancellation endpoint or immediate interruption of CPU inference. The lock and model references remain owned until the computation returns. Queued disconnected requests can also continue; there is no hard inference budget or upload deadline.

Automated offline tests use model doubles over actual local TCP connections to hold cross-encoder inference at a known barrier. They verify the client has disconnected while the server still reports one active search, release inference, and verify later unchanged responses. Throwing inference returns a safe 503 without stack/path/query details, releases capacity and permits reuse. A real Uvicorn shutdown while work is active must return that request's 200 response and run lifespan cleanup. Missing dataset/index/model/dependency, initialization failure, invalid inputs, queue limits and no-download health are also covered by the existing API suite.

The shutdown admission defect found in Phase 5 is fixed: `close()` signals stopping before waiting on the work lock. New and already queued searches receive 503; an active search retains its resources and completes before cleanup. Status distinguishes `stopping` from `stopped`. A regression test coordinates active, queued, closing and new work without relying on inference speed. No algorithm, cache identity or frozen rule changes are involved.

## Artifacts and checks

Every execution creates a new `experiments/results/phase6/phase6-.../` directory. Required outputs:

- `load_test_summary.json`: per-level attempted/successful throughput, counts, statuses and separate latency distributions.
- `latency_by_concurrency.csv`: per-level success/failure/timeout counts and both latency distributions.
- `method_latency.csv`: per-level/per-method counts, successful latency and status distributions.
- `resource_usage.csv`: timestamped native memory/stage samples.
- `error_summary.json`: every non-success plus timeout/cancellation probes and sampler errors.
- `correctness_checks.json`: direct/before/during/after checks and documented tolerance.
- `phase6_report.json`: status, command, configuration, hardware/software, source hashes, summaries, initialization counts, memory stages and limitations.

Additional outputs include raw requests/hits, direct rankings, workload, before/after health, logs, protected-file hashes, latency/concurrency and memory PNGs, and a uniquely named portable report for completed runs. Failed runs retain partial artifacts and explicit failure status; missing measured-level CSVs in an early failed run do not imply zero latency or successful execution. Full environment/check evidence is retained with the audit/run outputs.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_api_reliability.py -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --replay experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104
.\.venv\Scripts\python.exe -m app.evaluation.phase3 --replay experiments/results/phase3/phase3-20261031T065803Z-72804b2af31a
.\.venv\Scripts\python.exe -m app.evaluation.phase4 --replay experiments/results/phase4/phase4-20261031T071828Z-46027d7d4330
```

All ordinary tests use small fixtures/doubles and never download large models. The explicit load harness requires already prepared real resources. Figures derive from saved measured data; check success counts/rejections alongside percentile lines. There is no authentication, dashboard, Docker, cloud, new router, distributed infrastructure or paid service in this milestone. Recommended follow-up is to choose an explicit local latency/admission target and validate repeated mixed workloads with a deployment-appropriate warmup/cancellation policy; broader query/corpus coverage remains separate from serving reliability.


Final observed outcomes: [measured results](experiment_results.md#phase-6-mixed-query-api-reliability-2026-11-09), [portable report](../experiments/reports/phase6-20261109T153436Z-0e8e7412820d.json). All five requested concurrency levels ran; failures/rejections, memory growth, discarded controller prototypes and the invalid launcher-memory run remain explicit. Source and old-artifact checks, tests/replays, direct correctness and graceful cleanup passed. The host was not CPU-isolated, and regression checks/IDE shared the machine during portions of this run. Conditional successful throughput changes with rejected method mix and cannot establish scaling. A larger single-process soak can be requested explicitly, for example `--requests 1000 --clients 1 --warmup 40`, after choosing a memory/error budget; it takes substantially longer and was not performed in this milestone.


Phase6.1 now provides [duration-based same-process soak/idle diagnostics](soak_testing.md). Its completed long campaign, memory attribution limits and failed local budgets are recorded separately; it did not alter Phase6 algorithms or earlier measured outputs.
