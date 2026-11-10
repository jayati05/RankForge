# Phase 6.1: sustained API and memory investigation

This diagnostic campaign extends the Phase 6 workload without changing the API, retrieval algorithms, model adapters, cache identity, or frozen routing policy. It tests several fixed-concurrency stages in **one serving process**, with drained idle recovery and fresh correctness checks between stages. Graceful shutdown occurs after the campaign so the second soak can explicitly reuse the first process.

## Exact Windows PowerShell commands

Reuse prepared SciFact data, pinned local models/index and the existing virtual environment, from the repository root. No new dependency or model download is needed.

```powershell
# Functional smoke; too short for a stability conclusion
.\.venv\Scripts\python.exe -m app.evaluation.api_soak --duration 20 --clients 1,2 --stress-duration 10 --idle 5 --warmup 5 --sample-interval 1 --window 5

# Registered long campaign: 15-minute normal stages, 5-minute stress, 5-minute idle after each
.\.venv\Scripts\python.exe -m app.evaluation.api_soak --duration 900 --clients 1,2 --stress-duration 300 --idle 300 --warmup 60 --sample-interval 1 --timeout 30 --seed 42 --window 120 --rss-budget-mib 2048 --growth-budget 2 --rejection-budget 0.05 --stress-rejection-budget 1 --p95-budget-ms 5000
```

The long campaign requires about 51 minutes plus initialization, allocation snapshots, correctness requests and draining. `--duration` is seconds **per normal stage**; `--clients` selects sequential fixed levels; `--stress-duration 0` disables stress. `--idle`, `--warmup`, `--sample-interval`, `--timeout`, `--seed`, `--methods`, `--query-set`, window and budget flags are configurable. Longer durations are supported. HTTPX timeouts are I/O timeouts, not server CPU deadlines. Once a stage's submission deadline arrives, active requests finish; this overrun is recorded. No automatic retries hide rejections.

Default weights are `bm25:1,dense:1,hybrid:1,reranked:1,router:1`, with all five required. Whole seeded cycles are balanced; a duration cutoff can truncate the final cycle. The raw outcomes and per-method summary record the actual submitted/admitted mix. Reranking receives 20% of attempted full cycles, rather than dominating by request count; it can dominate CPU time. Normal and stress stages are separately labelled and assessed. A custom query file is a JSON array of unique string IDs/nonblank text, e.g. `[{"id":"001","text":"IL-6: 2020?"}]`; original text is preserved. The frozen tokenless-query BM25 fallback is checked against the policy, not incorrectly assumed to be hybrid for every query.

## Pre-registered local investigation budgets

| Budget | Normal stages | Stress stage |
| --- | --- | --- |
| Peak process working set/RSS | <=2048 MiB | <=2048 MiB |
| Last-window RSS slope | <=2 MiB/min | <=2 MiB/min |
| HTTP 429 fraction | <=5% | <=100%, intentionally permits overload rejection |
| Timeouts, client cancellations, other errors | zero | zero |
| Successful-request p95 | <=5000ms | <=5000ms |

These are configurable **local research targets**, chosen before the long measurements, not universal production thresholds. A completed run can fail budgets. Missing/short analysis windows yield an indeterminate growth-budget result, not a pass. The stress rejection allowance is not a claim that rejected requests were served. All-attempt and success-only latencies, statuses, rejection rates, timeouts and cancellations remain explicit.

## Protocol and process identity

The default eight-query public/authored pool is reused from Phase 6. It includes four exposed SciFact claims, keywords, a longer question, technical terms/numbers, punctuation and repetitions. No qrels determine scheduling or correctness. A separate direct-retrieval child produces the 40 query/method reference rankings and exact frozen policy decisions, then exits before the API child starts. The seeded cycle and query set/hash are saved.

The API child uses the original app factory/lifespan and existing CPU models/cache. Private atomic control files are used only by the benchmark process; there is no new HTTP administration endpoint. Stdin is not read by a controller thread, avoiding the Windows native-import issue found in Phase 6. Uvicorn's **serving PID** is taken from its startup log and recorded separately from the Windows virtual-environment launcher. Allocation snapshots independently report the actual PID. Health before initialization must show zero initializations.

Models/indexes are initialized once without query inference through a benchmark-only control command. Native memory is marked before and after initialization. Python tracing, if enabled, starts **after** model loading with one frame per trace; use `--no-trace` for an explicitly untraced campaign. A duration-based warmup follows. An initial full correctness workload precedes the first diagnostic baseline snapshot; the measurement stage begins after these operations.

Each stage submits closed-loop requests at fixed concurrency until the duration deadline, then drains active work. Every successful response is checked against ordered direct IDs/ranks and scores (absolute tolerance <=1e-5, relative tolerance zero); selected routing strategies and canonical policy identity must match. After draining, a diagnostic snapshot is taken, followed by an idle period with **no submitted HTTP workload**. At its end, memory is sampled before further diagnostics/correctness requests. Recovery-check activity has its own stage label and is excluded from idle-window analysis. All query/method pairs are checked again; health and one-attempt/one-initialization counters are verified. Stages continue without restarting the server; final stop runs original lifespan cleanup and checks exit0/service-stop logs.

## Memory evidence and its limits

The monitor samples process working set/RSS, cumulative OS peak working set and private commit, plus native available/total system memory where supported. Windows uses `GetProcessMemoryInfo` and `GlobalMemoryStatusEx`, without WMI or an added package. Linux uses `/proc`; other platforms have not been validated. Samples are synchronized so periodic and stage-marker reads cannot race their system-memory fields. One-second sampling can miss transient peaks; OS peaks are lifetime values.

For each soak, the analysis reports whole-stage and late-window least-squares RSS slopes in MiB/min, early/late window means, observed span, peak, end memory, and idle-end/change. Default comparison windows are the first/last 120 seconds; classification requires at least two complete windows and three late samples. A slope inside the configured tolerance is described as approximately flat **in the observed late window**, not proof of bounded lifetime memory.

Tracemalloc snapshots before/after workloads and idle record traced current/peak bytes, tracer overhead, top allocation sites and changes from the first snapshot. They omit pre-tracing startup allocations and cannot observe all NumPy/PyTorch/native library memory. Diagnostic snapshots also record tracked-object counts (including request/response/task/tensor-related classes), Python thread count, event-loop task classes, logical model parameter/buffer bytes, dense index bytes and resource counters. Native extension metaclasses can expose non-string module descriptors; the census handles them safely. Object lists/snapshot comparisons are temporary except for one intentionally retained baseline snapshot. No forced garbage collection, allocator trimming or cache clearing is performed.

Interpretation distinguishes:

- Model/index bytes: directly measured logical tensor/array sizes, separate from Python/import/runtime overhead.
- Temporary workload memory: load peaks and recovery observations; RSS alone does not identify particular tensor allocations.
- Native allocator/workspace retention: a plausible explanation for elevated RSS with small traced changes and bounded tracked objects, not automatically a confirmed cause.
- Potential application-object retention: investigate sustained object/traced-allocation growth and its allocation sites before naming a leak.
- Confirmed leaks: require stronger retention/allocation evidence than a rising RSS line or one campaign provides.

Diagnostic census/snapshot operations themselves allocate temporary Python objects and retain one baseline trace snapshot. They run outside timed workload/idle intervals, but their allocator effects can remain. Tracing changes performance and memory; the results must not be directly equated with untraced Phase 6 timing or production limits. No native allocator profiler or full tensor lifetime profiler is claimed.

## Artifacts, failures and verification

Each run creates `experiments/results/phase6_1/phase6_1-.../` and retains partial/failed runs. Required files are `soak_summary.json`, `memory_timeseries.csv`, `latency_timeseries.csv`, `request_outcomes.csv`, `memory_analysis.json`, `correctness_checks.json`, `lifecycle_summary.json`, and `phase6_1_report.json`. Extra files include progress/checkpoint files, workload/query-set, direct references, allocation diagnostics, server/direct logs, source hashes, environment and final verification evidence. Request outcomes are streamed/flushed as attempts complete; finalizers save collected partial samples on ordinary interruption.

Figures show memory over campaign time, all-attempt latency over stage time, success/non-success by method, and idle recovery. All statistics and plots derive from saved measurements. Success latency excludes rejected requests while rejection fractions and complete outcomes remain present. Trace overhead, small tails, runtime/scheduling variation, one query pool and one host limit extrapolation. A research budget pass is separate from API readiness or production certification.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --replay experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104
.\.venv\Scripts\python.exe -m app.evaluation.phase3 --replay experiments/results/phase3/phase3-20261031T065803Z-72804b2af31a
.\.venv\Scripts\python.exe -m app.evaluation.phase4 --replay experiments/results/phase4/phase4-20261031T071828Z-46027d7d4330
```

No product memory-management optimization is justified by an RSS increase alone. The existing API cancellation limit remains: a client disconnect/timeout does not immediately stop CPU inference; serialization stays held until work returns. Steady soaks do not deliberately inject cancellation, and count any observed cancellations separately. Containerization readiness must be scoped to observed workloads/budgets and revalidated on the target OS/container limits; production readiness, Docker, new algorithms/policies and deployment are outside this milestone.


## Measured campaign and controller fix

The [completed exact report](../experiments/reports/phase6_1-20261109T162840Z-f2dea9f54ecb.json) records all three long stages and idle periods in one serving process. See [experiment results](experiment_results.md) for actual latency, memory slopes, budget failures, allocation interpretation, reference probe and retained failed/partial runs. Completion is explicitly separate from passing budgets; container readiness under the registered normal targets was not established.

A native module-descriptor census failure and a transient Windows control-file read denial were found in preliminary runs. The benchmark-only census now handles non-string descriptors; the controller retries transient read failures and writes replies atomically, while its parent retries transient/partial reply reads. Regression tests cover both. No product memory code was changed. The final campaign's transient-read count was zero, so fault injection provides the specific retry-path verification.

The original `memory_analysis.json` idle delta is workload-end to idle-end and includes the intervening stopped diagnostic. Additional `allocation_interpretation.json` metrics compare actual idle-start and idle-end markers:0/0/-9.680MiB. Use the latter for recovery during idle; do not mislabel diagnostic allocation as idle workload growth. Late-window private-commit slopes and the limitations of tracing started after model initialization are also recorded.

The optional `scripts/soak_object_probe.py` is a separate authored-six-document BM25 real-HTTP diagnostic, not part of the measured real-model campaign. It observes request referrers after rejected requests/idle and subsequent ordinary successful traffic, without forcing GC or synthetic allocation pressure. Its verified command/output/source hash is linked from the report. It found collectible references, not a definitive allocation root for all native or application memory.
