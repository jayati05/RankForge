# Optional MLflow tracking (Phase 7B)

RankForge keeps local experiment files as the canonical reproducible outputs. Tracking runs after those files have been saved and never enters a retriever, model adapter, metric function, API request or frozen routing decision. No autologging, model registry or weight upload is enabled.

## Architecture and local server

The optional client is `mlflow-skinny==3.5.1`. The API and ordinary offline installation do not require it. A separate, digest-pinned official MLflow 3.5.1 image runs through the existing Compose `tracking` profile. One worker uses SQLite on the backend volume and a separate artifact volume; uploads/downloads travel through the tracking server's HTTP artifact proxy. Clients need no access to the container filesystem.

Windows PowerShell, repository root:

```powershell
# Existing environment; do not recreate it.
.\.venv\Scripts\python.exe -m pip install -e '.[tracking]' --no-build-isolation --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip check
docker compose --profile tracking up -d --no-build --wait --wait-timeout 120 mlflow
Invoke-WebRequest http://127.0.0.1:5000/health -UseBasicParsing
# Open http://127.0.0.1:5000 in your browser.
docker compose --profile tracking logs --tail 30 mlflow
```

Default tracking URI and UI: **http://127.0.0.1:5000**. Change the published port before startup and use that same URI in the client:

```powershell
$env:RANKFORGE_MLFLOW_PORT='5500'
docker compose --profile tracking up -d --wait mlflow
$env:MLFLOW_TRACKING_URI='http://127.0.0.1:5500'
```

Compose variables: `RANKFORGE_MLFLOW_PORT`5000, `RANKFORGE_MLFLOW_MEMORY_LIMIT`1g, `RANKFORGE_MLFLOW_CPU_LIMIT`2. Backend URI is `sqlite:////mlflow/backend/mlflow.db`; artifact destination is `/mlflow/artifacts`. Default persistent volumes are `rankforge-local_mlflow_backend` and `rankforge-local_mlflow_artifacts`. Both survive container restart/recreation and ordinary Compose down. Do not use `down -v` if you want to retain tracking data. Stop the server before making a consistent SQLite/artifact backup. This single-workstation SQLite design is not a concurrent distributed tracking deployment.

The port is published only on host loopback. Allowed Host headers are localhost,127.0.0.1 and the Compose service name; host validation remains enabled. There is no authentication in this trusted local setup, so other users/processes with local access can access the service. Public binding/authenticated multiuser hosting is outside this phase. The RankForge API has no dependency on the tracking service or its volumes and receives no tracking URI. Starting the default Compose setup does not automatically start this profile.

## New experiments and disabling tracking

```powershell
# Authored synthetic plumbing test: real computation, not public-data quality evidence.
.\.venv\Scripts\python.exe -m app.evaluation.benchmark --dataset data/smoke --split test --k 10 --output experiments/results/tracked-smoke --tracking --tracking-uri http://127.0.0.1:5000 --tracking-experiment rankforge
# Public BM25 evaluation; existing dataset preparation/license rules still apply.
.\.venv\Scripts\python.exe -m app.evaluation.benchmark --dataset data/public/scifact --split test --max-queries 25 --seed 42 --k 10 --output experiments/results/tracked-scifact --tracking
# Optional flags also work on comparison, phase3 and phase4 CLIs.
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --max-queries 25 --seed 42 --require-cache --tracking
```

Omit `--tracking` to disable it. A URI flag/environment variable alone does not enable tracking. Disabled tracking creates no receipt, instantiates no MLflow client and requires no server or client package. Computation, saved result formats and replay mathematics are unchanged. New runs naturally record the current source hashes/software environment, including the added adapter code; this does not rewrite old metadata.

If tracking is enabled but unavailable, ordinary evaluation still succeeds and saves its outputs. It emits a warning plus a JSON receipt with failed status; the MLflow run ID is printed/persisted when available. Receipts live in ignored `experiments/tracking/*.json`, outside canonical source runs. Retry registration later without recomputing inference. The explicit historical-import CLI exits nonzero when registration fails, distinguishing a failed import command from a successfully saved evaluation.

The client defaults to a 5 second HTTP request timeout and zero HTTP retries; these can be explicitly configured through MLflow's environment settings. These are per-request limits, not a hard deadline for a whole upload. The pinned SDK's proxied artifact implementation consults the fluent tracking URI; the adapter scopes/restores that URI under its own lock for artifact operations. It does not leave fluent state changed. Do not concurrently switch fluent URIs from unrelated code in the same process. The adapter suppresses SDK run-link printing by default because its emoji failed under Windows cp1252; the receipt provides run IDs without requiring console encoding changes.

## Historical registration

Imports read saved files only; no retriever/model is instantiated and no quality metrics are recomputed. Supported inputs: Phase 1 BM25 run directories, Phase 2 comparison directories/reports, Phase 3 routing directories/reports, Phase 4 directories/reports and Phase 6/6.1/6.2/7A operational reports. Incomplete/failed executions are rejected; completed experiments with failed resource budgets remain valid measured operational reports, with limitations preserved.

```powershell
.\.venv\Scripts\python.exe -m app.tracking.mlflow experiments/reports/phase2_scifact_full.json --tracking
.\.venv\Scripts\python.exe -m app.tracking.mlflow experiments/reports/phase3_scifact.json --tracking
.\.venv\Scripts\python.exe -m app.tracking.mlflow experiments/reports/phase4_nfcorpus.json --tracking
.\.venv\Scripts\python.exe -m app.tracking.mlflow experiments/reports/phase6_1-20261109T162840Z-f2dea9f54ecb.json --tracking
.\.venv\Scripts\python.exe -m app.tracking.mlflow experiments/reports/phase6_2-repeat-20261109T194018Z-c8b6ba59.json --tracking
.\.venv\Scripts\python.exe -m app.tracking.mlflow experiments/reports/phase7a-20261110T063328Z-3ae104cc.json --tracking
# Repeating the same completed import returns its existing parent ID.
.\.venv\Scripts\python.exe -m app.tracking.mlflow experiments/reports/phase2_scifact_full.json --tracking
# Verify an expected primary-source hash before any server writes.
$sha=(Get-FileHash experiments/reports/phase2_scifact_full.json -Algorithm SHA256).Hash.ToLowerInvariant()
.\.venv\Scripts\python.exe -m app.tracking.mlflow experiments/reports/phase2_scifact_full.json --tracking --expect-sha256 $sha
```

Existing replay commands also accept `--tracking`; they first verify the local replay and then register it as historical. Replay without the flag remains independent.

## Runs, parameters and metrics

`--tracking-experiment rankforge` creates two experiments: `rankforge/evaluation` and `rankforge/operational`. Each source has a parent run with artifacts and one child per method/condition/concurrency row. Parent names include new/historical mode and original run ID. Child names identify method and condition. Receipts contain the parent and newly created child IDs.

Parents retain source paths/references, primary-file SHA256, selected-artifact hashes, original metadata and timestamps. An available top-level original timestamp sets MLflow start/metric time and is also tagged. If no supported top-level timestamp exists, MLflow creation time is import time; original nested dates remain in the metadata snapshot. Import end time is registration completion, not invented original experiment duration. Historical mode never implies fresh inference.

The current schema is `rankforge-tracking-v2`. Its duplicate identity covers schema, primary source hash, selected-artifact hashes and normalized parameters/metrics. Changed source/normalization gets a new identity. Repeating a completed identical import in the same experiment/mode reuses the parent. Failed partial parents are retained and not reused. Imports are intended to run sequentially: MLflow search/create is not a distributed uniqueness transaction, so simultaneous importers may race. Using a report wrapper versus a raw directory can produce different source identities; use a consistent entrypoint.

| Information | MLflow representation |
| --- | --- |
| Dataset, version/provenance, split, corpus/query/judgment counts | Named parameters when recorded |
| Method,k,seed,BM25constants, RRF/depth/pool,batch/threads | Parameters copied from saved configuration |
| Model IDs/revisions, library versions, device/platform | Parameters when recorded; full metadata artifact |
| Recall/Precision/MRR/nDCG@k | `eval.recall_at_k`, `eval.precision_at_k`, `eval.mrr_at_k`, `eval.ndcg_at_k` |
| Offline mean/p50/p95/p99 latency | `eval.*_latency_ms`, only where saved |
| Measured setup/model/index/embedding construction | `setup.*`, only where saved |
| Service counts, throughput, success/all-attempt percentiles | `ops.*`, never quality metrics |

Missing/nonfinite metrics are omitted, never filled with zero. Missing parameters are omitted rather than inferred from another population. For example, Phase 3 held-out judgment counts are not copied from all 300 SciFact queries. Setup metrics describe the whole experiment context, are repeated on child rows for inspection and must not be added as independent per-method costs. Old percentiles omitted for small samples remain absent.

Container operational parameters separate actual container Python/image identity from `report_host_python`; host Windows metadata must not be mistaken for the Linux runtime. Quota, condition, explicit BLAS override and client count are preserved. Available image package versions are copied from its saved freeze; other runtime details remain in the source report. Operational reports lacking dataset/device/version fields do not receive guessed values.

## Artifacts and comparisons

The allowlist includes saved metadata, summaries, per-query/system CSVs, failure summaries/cases, routing decisions/query features, provenance/configuration and selected plots. It excludes raw corpus/query/qrels files, request-result document payloads, model weights, indexes, caches, logs, environment files and temporary files. Symlinks/out-of-root paths are excluded; individually selected files above 20MiB are skipped. JSON/CSV snapshots redact common credential fields, environment assignments and URI credentials; source originals remain unchanged. This is a bounded experiment adapter, not a general-purpose secret scanner. Inspect inputs before explicitly importing reports from other projects.

`source_manifest.json` records original-source hashes even when a sanitized upload has different bytes. `source_report.json` is a sanitized primary JSON report. Uploaded artifacts are supplementary copies; local canonical files plus the existing replay commands remain the reproducibility authority.

In the MLflow UI, select the correct evaluation/operational experiment, expand a completed source parent, and compare its method children. Filter `rankforge.comparison_group` to one source bundle for same-scope comparisons. Filter parents by `rankforge.registration_complete=true` and `rankforge.schema=rankforge-tracking-v2`; retained development trials/failed partial runs are not final verification evidence.

300-query SciFact Phase 2, previously exposed internal 60-query Phase 3 and 323-query NFCorpus Phase 4 differ in scope, judgments and distributions. Cross-dataset panels are descriptive, not equivalent paired experiments. Native Windows versus Docker Linux also differs in OS, dependencies, quotas, caches and timing; use condition/client/platform parameters and source methodology, not a pooled latency average. Operational HTTP timing is not offline retrieval timing. This integration adds no new relevance scores or generalization claims.

## Verification and recovery

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
# Real server, history/equality/dedup, UI HTML/assets, persisted artifacts,
# restart/recreation, outage-tolerant smoke and five API methods during outage.
.\.venv\Scripts\python.exe scripts/verify_mlflow.py
docker compose --profile tracking stop --timeout 30 mlflow
docker compose --profile tracking up -d --no-build --wait mlflow
```

The integration verifier leaves MLflow healthy for UI inspection and the API stopped after its probe. It preserves named volumes and never issues `down -v`. Each execution creates unique Phase 7B artifacts/reports. Unit tests use client doubles, authored fixtures and imports blocked to prove offline independence; they do not require MLflow installed. Browser rendering beyond HTTP HTML/JavaScript retrieval is not claimed. See [measured results](experiment_results.md) for actual run IDs/counts, failures and limits.

Official references: [self-hosted server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/), [artifact proxy/storage](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/), [network protection](https://mlflow.org/docs/latest/self-hosting/security/network/), [official pinned image](https://github.com/mlflow/mlflow/pkgs/container/mlflow/552681943?tag=v3.5.1). No cloud, Kubernetes, production BLAS change, retrieval rewrite or weight/dataset upload is part of this milestone.
