# Local Docker service (Phase 7A)

Phase 7B follow-up: the [optional MLflow service](mlflow_tracking.md) uses the existing Compose `tracking` profile with separate persistent SQLite/artifact volumes. The API has no `depends_on` relationship or tracking dependency. `docker compose --profile tracking up -d --wait mlflow` starts only tracking; default API commands remain valid. This addition does not change the Phase 7A resource findings below.

This packages the existing single-worker CPU API. No cloud deployment or MLflow is included. Phase 6.2's failed growth/rejection budgets and inconclusive leak remain unresolved. A successful image build is not deployment readiness.

Completed validation: [Phase7A report](../experiments/reports/phase7a-20261110T063328Z-3ae104cc.json).204tests and840correctness checks passed; all six container stops and the separate active-inference/recreation probe passed. The560attempt workload had35HTTP429and no timeouts. Default C2rejection exceeded5%, and2CPUreranked p95exceeded5000ms. The393.46MiB image built in134.518s with cached base layers. This supports monitored local MLflow integration work, not stable deployment. Full per-method/resource tables and experimental BLAS limitations are in experiment_results.md.

## Prepare resources and build

Use Docker Desktop with Linux containers. Commands below run in Windows PowerShell from the repository root. Reuse the existing prepared dataset, pinned model snapshots and frozen `experiments/configs/phase4-transfer` capsule. The full preparation commands and license notices are in [the API guide](api_service.md) and README. The API never downloads models or builds an index through HTTP.

```powershell
docker version
docker compose version
# Only if missing; these existing CLIs validate/reuse prepared files.
.\.venv\Scripts\python.exe -m app.data.download
.\.venv\Scripts\python.exe -m app.data.models --dense-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --rerank-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
docker compose --progress plain build
# Explicit offline Linux index preparation. Does not overwrite the Windows index.
$indexMount = (Join-Path (Get-Location) 'indexes').ToString() + ':/prepared'
docker compose run --rm --no-deps --volume $indexMount api python -m app.retrieval.dense --dataset /resources/dataset --model-path /resources/models/dense --cache /prepared/scifact-minilm-linux.npz
docker compose up -d --no-build --wait --wait-timeout 90
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 8
$body = @{query='vitamin D'; methods=@('bm25','dense','hybrid','reranked','router'); top_k=10} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/compare -Method Post -ContentType application/json -Body $body | ConvertTo-Json -Depth 12
docker compose logs --tail 40 api
docker compose stop --timeout 60
docker compose start --wait --wait-timeout 90
docker compose down --timeout 60
```

The Linux CPU wheel reports `torch==2.9.0+cpu`; the Windows installation reports `2.9.0`. Dense cache validation includes library identity. Keep separate `scifact-minilm-linux.npz` and `scifact-minilm.npz` caches; never edit cache metadata to bypass validation. The Linux cache is generated once with the same corpus, model, batch16 and torch4 settings. A stale index fails until explicitly rebuilt at a new path. No prior experiment cache is replaced.

## Image and storage

One stage uses Python3.11.14 slim Bookworm, pinned by immutable image-index SHA256. No compiler or system package is installed. CPU-only torch and all runtime dependency versions are pinned in `docker/requirements.lock`; `pip check` runs at build time. This is a Linux/amd64 image. It deliberately uses a patch-version update from the Windows3.11.5 runtime, and separate pinned Linux transitive versions; compare measured results rather than assuming cross-platform binary equivalence. Version pins do not claim byte-identical rebuilds or perpetual availability of external wheel servers.

Only Python source and the dependency lock enter the allowlisted build context. There is no `COPY . .`, local venv, Git history, dataset, experiment result or weight copy. UID/GID10001 runs an exec-form Python entrypoint that checks required mounted file presence, then replaces itself with `python -m app.api`. Uvicorn owns PID1, receives SIGTERM directly, drains active work and runs existing lifespan cleanup. One worker preserves existing queue/cache semantics. No measurable benefit from a second build stage was established, so none is added.

| Host default | Container path | Mode |
| --- | --- | --- |
| data/public/scifact | /resources/dataset | read-only bind |
| models | /resources/models (dense, reranker) | read-only bind |
| indexes | /resources/indexes | read-only during serving |
| experiments/configs/phase4-transfer | /resources/policy | read-only bind |

Compose refuses absent bind-source paths instead of creating empty directories. On Linux ensure UID10001 can traverse/read the mounts and write the index directory only for the explicit offline preparation command. No broad chmod/chown is performed. Snapshots and indexes persist when the container is removed. In-memory resources do not persist: lazy initialization repeats once per new serving process, while disk artifacts are reused. Logs go to stdout/stderr. The service generates no durable runtime files; `/tmp` is an ephemeral128MiB tmpfs, not persistent storage. Evaluation artifacts are written by the host verifier. Container rootfs is read-only; capabilities are dropped and privilege escalation disabled. Host mount paths are never accepted from API requests.

## Configuration

The existing `RANKFORGE_DATASET_DIR`, `DENSE_MODEL_DIR`, `RERANK_MODEL_DIR`, `DENSE_INDEX`, `POLICY_DIR`, `HOST`, `PORT`, `DEFAULT_TOP_K`, `THREADS`, `BATCH_SIZE`, queue and candidate settings remain environment based. `DEFAULT_METHOD` now configures omitted `/search` methods (default BM25); explicit request methods and null validation are unchanged. `LOG_LEVEL` defaults to info and accepts debug/info/warning/error/critical. The native service still defaults to loopback, BM25, k10, torch4, batch16 and unchanged admission limits. Compose binds inside the container to0.0.0.0 and publishes only host127.0.0.1.

Compose-specific variables (set before running Compose):

| Variable | Default |
| --- | --- |
| RANKFORGE_PUBLISHED_PORT / RANKFORGE_PORT | 8000 / 8000 |
| RANKFORGE_DATASET_SOURCE | ./data/public/scifact |
| RANKFORGE_MODELS_SOURCE | ./models |
| RANKFORGE_INDEX_SOURCE / RANKFORGE_INDEX_FILENAME | ./indexes / scifact-minilm-linux.npz |
| RANKFORGE_POLICY_SOURCE | ./experiments/configs/phase4-transfer |
| RANKFORGE_MEMORY_LIMIT / RANKFORGE_CPU_LIMIT | 4g / 4 |

Compose supports `.env` substitution; local `.env*` files are excluded from the image and should never contain committed secrets. Do not confuse host source paths with absolute Linux application paths. Change the index filename when selecting another corpus. Frozen policy integrity is still checked against the mounted capsule and unchanged source implementation.

Health checks call the existing `/health`, require JSON `status=ok`, and use stdlib Python rather than installing curl. Health performs no model initialization and is not all-method readiness. Container-only preflight fails immediately when required corpus/index/model manifests/policy files are absent. Corrupt-but-present resources still fail safely on first use; full validation and warmup require explicit searches. No auto-restart policy hides crashes or OOM exits. Compose grants60s before SIGKILL; extremely slow active inference can still exceed that deadline.

BLAS settings are not set by the Dockerfile or default Compose file. Linux runtime defaults may differ from Windows. Explicit experiments use a second Compose file:

```powershell
$env:RANKFORGE_EXPERIMENT_BLAS_THREADS='1'
docker compose -f compose.yaml -f compose.blas-experiment.yaml up -d --no-build
docker compose -f compose.yaml -f compose.blas-experiment.yaml down --timeout 60
Remove-Item Env:RANKFORGE_EXPERIMENT_BLAS_THREADS
```

This changes OpenBLAS only. It does not set OMP/MKL or change production defaults. Record actual runtime library settings before interpreting it as active parallelism. For a standalone `docker run`, native `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` can be passed explicitly with `-e`; they are not implicit recommended settings.

## Integration verification and interpretation

Ordinary tests do not need Docker, downloaded datasets or real models. The explicit integration script creates a unique Compose project and Phase7A output directory, builds the image, prepares/reuses the Linux index, checks required-file failure, starts a fresh local API and sequential container configurations, restarts each container, then cleans up only its own project.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
# Small plumbing run, plus explicit experimental control
.\.venv\Scripts\python.exe scripts/verify_container.py --requests 40 --clients 1,2 --limits 4g:4,2g:2 --blas-threads 1
# Active-inference SIGTERM and complete container recreation, separately
.\.venv\Scripts\python.exe scripts/verify_container_lifecycle.py
# Larger pass; no BLAS override
.\.venv\Scripts\python.exe scripts/verify_container.py --requests 200 --clients 1,2,4 --limits 4g:4,2g:2
```

The eight exact public/authored queries, equal-five-method distribution, seed42, k10 and existing HTTP batch/aggregation implementation are reused. Local direct inference finishes before HTTP measurements. All40 query/method pairs and `/compare` are checked before each measured pass; every successful load response is checked too. IDs/ranks must match exactly; scores use the existing absolute1e-5 tolerance and frozen policy identity must match. Errors/rejections/timeouts are retained, without retries. A larger score delta fails strict equivalence even if rankings match. First-use pipeline setup includes dependencies and inference; API initialization components are recorded separately and must not be summed recursively. Model warmup is an explicit five-method pass.

`--requests` applies per concurrency level and again after each restart. Each configured limit uses a fresh container; restart resets in-memory counters. Default resource limits are safety ceilings for this experiment, not new performance budgets. Docker CPU quota limits scheduling, not the configured torch/BLAS thread count. Cgroup-v2 `memory.current`, `memory.peak`, `memory.events`, `cpu.max` and `cpu.stat` are sampled separately from `/proc/1` RSS/peak/thread counts, about2s plus observer duration. Cgroup totals include page cache, health/observer processes and tmpfs; they are not Windows private commit. Docker Desktop's Linux PID cannot be read as a host Windows PID. Local Windows process RSS/private commit are sampled from the verified Uvicorn serving PID. Docker Desktop host overhead is not attributed to this single container. Sampling can miss peaks; cgroup lifetime peak includes more than measured workload.

Required JSON/CSV artifacts, raw results, configuration, commands, logs, version/engine information, image ID/digests and source/resource hashes remain under `experiments/results/phase7a/phase7a-*`. Empty CSVs retain headers on early failure; missing measurements are not zeros. Three figures compare measured latency, process RSS and startup/first-use time. See [measured outcomes](experiment_results.md) for the actual completed scope and limitations. Interrupted/failed runs are preserved. No leak, stable lifetime memory, production capacity or cloud readiness conclusion follows from a short packaging test.

Derive CPU utilization, separate process/cgroup peaks, request outcomes and initialization-counter checks from a completed run without starting Docker again:

```powershell
.\.venv\Scripts\python.exe scripts/summarize_container.py experiments/results/phase7a/REPLACE_WITH_RUN_DIRECTORY
```

CPU percentages use one core=100%; a quota of four CPUs can approach400%. These are measured cgroup CPU deltas over sampled wall time, including auxiliary health/observer processes. The initial verifier CSV stored nested counters as Python dictionary text; the summarizer also accepts current JSON-encoded counters and preserves the original observations.

Some observed process RSS values exceed their cgroup's charged memory. These counters measure different things: shared resident pages need not be charged to the accessing process's current cgroup. The [kernel's memory ownership documentation](https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory-ownership) explains that shared pages may retain another cgroup's charge. This is a possible contributor here, not a measured page-by-page attribution. Do not subtract these counters to infer missing memory. Samples are sequential, and OS file caches were not flushed between configurations.

Compose limits, health and stop behavior follow the [official service reference](https://docs.docker.com/reference/compose-file/services/). The [official Python image](https://hub.docker.com/_/python) is pinned in the Dockerfile by digest; the actual local image ID is retained separately in the build report.
