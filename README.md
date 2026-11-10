# RankForge

Phase 7B adds [optional MLflow tracking](docs/mlflow_tracking.md). Start it with `docker compose --profile tracking up -d --wait mlflow`, then open `http://127.0.0.1:5000`. Install the optional client with `.\.venv\Scripts\python.exe -m pip install -e '.[tracking]' --no-build-isolation --index-url https://pypi.org/simple`. Add `--tracking` to existing evaluation CLIs, or register saved reports with `.\.venv\Scripts\python.exe -m app.tracking.mlflow SOURCE --tracking`. Offline evaluation and the API remain independent of MLflow; datasets and weights are excluded from tracking uploads.

Local container commands and measured limitations: [Phase 7A Docker guide](docs/docker_deployment.md).
With the existing prepared resources, run `docker compose --progress plain build`, prepare the separate Linux index using the guide, then `docker compose up -d --no-build --wait`. Stop with `docker compose down --timeout 60`. Run `.\.venv\Scripts\python.exe scripts/verify_container.py --requests 40 --clients 1,2 --limits 4g:4,2g:2` for isolated integration verification. This is a local research container; Phase 6.2's resource-budget failures remain open.

Phase7A measured: **204tests and840correctness checks passed**. Image393.46MiB; measured build134.518s with cached base layers. Local/container workload525successes/35HTTP429 across560attempts, no timeouts; all container stops/restarts and active-shutdown/recreation probe passed. Default rejection/latency targets still fail; BLAS1was experimental only. [Measured report](experiments/reports/phase7a-20261110T063328Z-3ae104cc.json) and [complete results](docs/experiment_results.md) retain limits and all-method latencies. These are Phase 7A observations; optional MLflow tracking followed in Phase 7B. No cloud deployment was added.

An experimental search and ranking platform investigating when retrieval fails and how ranking quality trades off against cost. **Phases 0–7B are implemented:** validated data loading, BM25, ranking metrics, a synthetic smoke benchmark, public SciFact/NFCorpus evaluation, dense retrieval, hybrid RRF, cross-encoder reranking, frozen routing, a local API, Docker validation and optional MLflow tracking. Verified comparisons cover all 300 SciFact test queries; measured results are linked below. This is a local research project; failed resource budgets and inconclusive memory stability prevent a production-readiness claim.

From the repository root in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[api,dev]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m app.evaluation.benchmark --dataset data/smoke --split test --k 10
```

On Linux/macOS use `python3.11 -m venv .venv` and `.venv/bin/python` instead. Activation is optional. Runtime dependencies are pinned in pyproject.toml; the original environment is recorded in `docs/environment-freeze.txt` and the Phase 5 environment in `docs/environment-phase5-freeze.txt`.

Each benchmark creates a unique directory under `experiments/results/` containing summary/per-query CSVs, rankings JSON and run metadata with input SHA256 hashes and environment details. Results are local ignored artifacts. Never interpret smoke scores as real-world retrieval quality.

The public repository includes `.env.example`, the authored `data/smoke/` fixture, the frozen `experiments/configs/phase4-transfer/` capsule and selected representative JSON reports. Downloaded datasets, model weights, indexes, full run directories and MLflow stores stay local. Saved reports/capsule retain original provenance, including original machine paths; those paths are historical metadata, not portable setup instructions. Replay commands below require the corresponding local run directories and prepared datasets, which are not included in a fresh checkout. Links in historical documentation to full run artifacts or unselected reports refer to local-only evidence. See the retained [Phase 6.2 report](experiments/reports/phase6_2-repeat-20261109T194018Z-c8b6ba59.json) and [Phase 7B report](experiments/reports/phase7b-20261110T073341Z-12c807e3.json) for representative later milestones.

See [methodology](docs/methodology.md), [data provenance](data/README.md), [progress](docs/decisions.md), and [verified results](docs/experiment_results.md). The Failure Lab, exploratory routing and FastAPI service are implemented. [Local Docker packaging](docs/docker_deployment.md) is available; dashboard and CI remain future work. No license has been selected for project code; dependency and dataset licenses remain their respective owners'.


## Public SciFact benchmark (Windows PowerShell)

Use the existing `.venv` from installation above. The following download/import path was verified on this machine. It uses Windows certificate trust; Python's direct HTTPS download failed certificate verification here. No certificate verification is disabled.

```powershell
New-Item -ItemType Directory -Force data/public/archives | Out-Null
if (-not (Test-Path -LiteralPath data/public/archives/scifact.zip)) {
    Invoke-WebRequest -Uri 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip' -OutFile data/public/archives/scifact.zip -UseBasicParsing
}
.\.venv\Scripts\python.exe -m app.data.download --archive data/public/archives/scifact.zip

# Small query subset, with all 5,183 documents still searchable
.\.venv\Scripts\python.exe -m app.evaluation.benchmark --dataset data/public/scifact --split test --max-queries 25 --seed 42 --k 10 --output experiments/results/scifact

# Complete BEIR test split: all 300 queries
.\.venv\Scripts\python.exe -m app.evaluation.benchmark --dataset data/public/scifact --split test --k 10 --output experiments/results/scifact

# Offline checks; no public download needed
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

On hosts with working Python HTTPS trust, the portable download command is:

```powershell
.\.venv\Scripts\python.exe -m app.data.download
```

The downloader verifies the pinned official MD5 before staging any data, validates both splits with the existing loader, saves archive SHA256 and per-file hashes, and reuses an existing destination only after checking its recorded hashes. It refuses corrupt archives or changed datasets. If a PowerShell download is interrupted, retry using a fresh archive filename and pass that file to `--archive`; the importer will reject a partial file. Use `--destination` for another installation location. Downloaded data and archives stay in ignored `data/public/`; no dataset redistribution is part of this repository.

`--max-queries` selects a reproducible SHA256-ordered query subset from `--split`; `--seed` controls selection. Omitting the limit evaluates the entire split. Document IDs, query IDs, and judgments remain unchanged. Use `--split train` for future development work. The 25-query test subset is a plumbing check, not a development/validation split; all test results reported here used the original, untuned BM25 settings.

Full test-split results: Recall@10 **0.759833**, Precision@10 **0.083333**, MRR@10 **0.598321**, nDCG@10 **0.633399**. These are this implementation's measured baseline, not a reproduction of a published BEIR BM25 score. Unjudged documents receive zero gain for evaluation; they are not asserted to be irrelevant. See [full results and limitations](docs/experiment_results.md) and the [machine-readable report](experiments/reports/scifact_bm25.json).

SciFact attribution: Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020; BEIR: Thakur et al., 2021. The [BEIR SciFact card](https://huggingface.co/datasets/BeIR/scifact) lists [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/). Preserve attribution, the license link and change notices; adaptations distributed under this license must retain ShareAlike terms. This data license does not select a license for RankForge's code.


## Phase 2 local neural retrieval

Install the optional dependencies; the lexical-only install remains supported. These commands use Python 3.11 and require no GPU or paid service:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[api,neural,dev]'
.\.venv\Scripts\python.exe -m app.data.models --dense-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --rerank-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
.\.venv\Scripts\python.exe -m app.retrieval.dense --dataset data/public/scifact --cache indexes/scifact-minilm.npz --batch-size 16 --threads 4

# All four systems, same reproducible 25-query subset and full document corpus
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --max-queries 25 --seed 42 --require-cache

# Same frozen settings, all 300 BEIR test queries
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --require-cache

# Real-model synthetic smoke, kept separate with its own index
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/smoke --cache indexes/smoke-minilm.npz --candidate-depth 5 --rerank-candidates 3 --k 3 --output experiments/results/phase2-smoke
```

The model download command uses system certificate trust (`truststore`), keeping HTTPS verification enabled; this worked for both snapshots on this machine. Downloads resolve a full revision, resume the same pinned partial snapshot, and record file hashes in `rankforge_model.json`. Repeating the command verifies/reuses local files. Use `--dense-model`, `--rerank-model`, revision flags and separate `--dense-path`/`--rerank-path` destinations to select other compatible models. Existing model snapshots are never silently replaced. See model cards before changing models. Inference is local-only with remote custom code disabled.

`--batch-size`, `--threads`, `--candidate-depth`, `--rrf-constant`, `--rerank-candidates`, `--k`, `--max-queries`, and `--seed` configure the comparison. Defaults: CPU, 4 torch threads, batch 16, source depth 50, RRF constant 60, rerank pool 20, k=10. These settings were frozen before evaluation; use train/development data to change them. Never reuse the SciFact cache for smoke data. A stale or corrupt cache raises an error; choose a new `--cache` path to rebuild. `--require-cache` rejects a missing index. Models and indexes remain ignored local artifacts.

Dense retrieval precomputes 384-dimensional document vectors, normalizes vectors and queries, then ranks by cosine dot product with NumPy. Model input is original title + space + body, not the lexical tokenizer. The default dense model truncates at 256 wordpieces. RRF adds `1/(constant + rank)` for each source containing a document; it uses equal weights, deduplicates IDs and breaks ties by ID. Hybrid never adds raw BM25 and cosine scores. The cross-encoder scores only the top 20 hybrid candidates, truncating the query/document pair at 512 tokens and sorting raw logits. It cannot recover a document outside that pool. The pool must be smaller than the corpus; use the explicit smaller settings for smoke data.

Phase 2 runs use unique directories under `experiments/results/phase2/`. They save metadata/status, summary CSV/JSON, per-query CSV, all four rankings, pre/post reranking candidate pools, and disagreement analysis. Failed/interrupted runs retain partial artifacts and `status=failed`; they are not complete comparisons. Neural model initialization and index setup are outside query timing. All pipelines get one authored-text warmup and execute independently in a rotated order; hybrid/reranked latency includes their upstream retrieval. Percentiles are descriptive and omitted below 20 queries.

To verify a completed run without loading models, substitute its printed directory:

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --replay experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104
.\.venv\Scripts\python.exe -m app.evaluation.report --dataset data/public/scifact --run experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104 --output experiments/reports/phase2_scifact_full_replayed.json
```

[Phase 2 methodology and model limitations](docs/phase2.md) describes model licenses, truncation, cache identity and fair timing. Measured results are in [experiment results](docs/experiment_results.md). Ordinary pytest tests use controlled vectors and model doubles, with no model/data download.


Verified full Phase 2 results (300 queries):

| Method | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.759833 | 0.083333 | 0.598321 | 0.633399 | 18.285 | 30.848 |
| dense | 0.783333 | 0.088333 | 0.604725 | 0.645082 | 11.669 | 16.447 |
| hybrid | 0.822111 | 0.091000 | 0.637352 | 0.676569 | 29.284 | 43.285 |
| reranked | 0.834722 | 0.093000 | 0.668989 | 0.700349 | 944.082 | 1041.990 |

Reranking improved observed average quality over hybrid, with about 31.11 times its mean latency; it improved per-query nDCG for 65 queries and degraded it for 53. All candidate counts were 20. These are single-machine warmed CPU observations with model loading/index setup excluded, not general performance or significance claims. [Exact full report](experiments/reports/phase2_scifact_full.json) and [subset report](experiments/reports/phase2_scifact_25.json) retain metadata and selected observed improvements/regressions. The [results documentation](docs/experiment_results.md) records original indexing time, regressions, validation and limitations. Final checks: 93 tests passed; lint, formatting and dependency checks passed.


## Phase 3: Failure Lab and exploratory routing

The [Failure Lab](docs/ranking_failure_lab.md) provides evidence-backed, nonexclusive failure indicators, complete per-query comparisons, and label-free query diagnostics. The [router methodology](docs/query_aware_routing.md) documents the internal 180/60/60 split, development/validation selection, frozen policy, live evaluation and paired bootstrap intervals. This is an internal exploratory split of a previously inspected benchmark, not untouched external evaluation.

From the repository root in Windows PowerShell (reuse the existing SciFact data, models and index):

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[api,neural,analysis,dev]'
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --replay experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104
.\.venv\Scripts\python.exe -m app.evaluation.phase3 --dataset data/public/scifact --source experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104 --seed 42
.\.venv\Scripts\python.exe -m app.evaluation.phase3 --dataset data/public/scifact --replay experiments/results/phase3/phase3-20261031T065803Z-72804b2af31a
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

The Phase 3 execution creates a new unique directory. It inherits the source pipeline settings, requires the validated dense cache and local models, saves exact split IDs and the policy before evaluation, then measures five systems on the same 60 internal held-out queries. Do not rerun evaluation to tune against held-out outcomes. Replay performs no model inference. Use the existing model/data commands above if those artifacts are absent; no fresh download was needed for this run. Matplotlib is the optional analysis dependency; ordinary tests do not need to render figures or download models.

Required CSV/JSON artifacts and five figures are in `experiments/results/phase3/phase3-20261031T065803Z-72804b2af31a/`. The [portable measured summary](experiments/reports/phase3_scifact.json) retains policy, splits, provenance, metrics and examples outside the ignored results directory; full raw rankings remain in the run. Prior Phase 1/2 artifacts are untouched.

**Measured negative result:** validation selected the hybrid-only fallback. The router chose hybrid for 100% of internal held-out queries, invoked reranking 0%, and matched hybrid quality. It does not demonstrate adaptive routing benefit. Relative to always-reranking, it saved 96.48% mean latency while losing 0.050054 nDCG; the exploratory paired 95% nDCG interval includes zero. Sparse judgments, prior benchmark exposure and single-pass CPU timings limit conclusions.

| Method (same 60 queries) | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Mean ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.755556 | 0.078333 | 0.573353 | 0.611713 | 19.093 | 18.178 | 33.093 |
| dense | 0.758333 | 0.081667 | 0.567222 | 0.613868 | 13.807 | 11.479 | 19.563 |
| hybrid | 0.802778 | 0.085000 | 0.593419 | 0.639072 | 31.107 | 29.910 | 44.030 |
| reranked | 0.858333 | 0.091667 | 0.647057 | 0.689126 | 896.722 | 906.554 | 983.601 |
| router | 0.802778 | 0.085000 | 0.593419 | 0.639072 | 31.550 | 29.604 | 51.859 |

Final checks: 104 tests passed; lint/formatting/dependencies passed; Phase 2 replay verified 1,200 rows and Phase 3 replay verified 300 rows. See [measured results and limitations](docs/experiment_results.md).


## Phase 4: external NFCorpus validation

The exact saved Phase 3 policy was frozen before NFCorpus preparation. No NFCorpus labels were used for tuning, inference-time features or routing. The official BEIR archive passed its published checksum and remains preserved locally. The [external-validation methodology](docs/external_dataset_validation.md) records license, provenance, split counts, timing and leakage controls.

Observed NFCorpus test scope: **3,633 documents, 323 queries, 12,334 judgments**, including grades 1 and 2. All official test IDs and the complete corpus were retained. The metric definitions and model/pipeline settings are unchanged from prior phases. Unjudged documents receive zero evaluation gain, without being asserted irrelevant.

Windows PowerShell, repository root (reuse the installed models and virtual environment):

```powershell
# Freeze must precede external-data import
.\.venv\Scripts\python.exe -m app.evaluation.transfer_policy
.\.venv\Scripts\python.exe -m app.data.nfcorpus
.\.venv\Scripts\python.exe -m app.retrieval.dense --dataset data/public/nfcorpus --cache indexes/nfcorpus-minilm.npz --batch-size 16 --threads 4
.\.venv\Scripts\python.exe -m app.evaluation.phase4

# Verify the completed run without loading models
.\.venv\Scripts\python.exe -m app.evaluation.phase4 --dataset data/public/nfcorpus --replay experiments/results/phase4/phase4-20261031T071828Z-46027d7d4330
```

Each execution creates a new Phase 4 directory. It does not replace the Phase 3 policy or previous results. Checksum-verified offline archive import and Windows HTTPS fallback commands are documented in the methodology. No model was downloaded again. Ordinary tests use small authored fixtures and require no external data/model download.

| Method (323 NFCorpus queries) | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Mean ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.140725 | 0.211455 | 0.499257 | 0.294681 | 6.326 | 5.927 | 12.922 |
| dense | 0.154988 | 0.243344 | 0.507654 | 0.317272 | 23.181 | 16.555 | 55.027 |
| hybrid | 0.154521 | 0.238080 | 0.538718 | 0.324066 | 27.903 | 22.956 | 58.607 |
| reranked | 0.161816 | 0.252322 | 0.575036 | 0.351809 | 1896.101 | 1378.073 | 4301.621 |
| router | 0.154521 | 0.238080 | 0.538718 | 0.324066 | 26.119 | 21.772 | 57.153 |

The frozen router selected **100% hybrid**, **0% BM25/reranking**, and matched hybrid quality. It saved **98.62%** mean latency versus always-reranking, losing **-0.027743 nDCG**. This preserves a fixed quality/latency compromise, not an adaptive routing advantage. Hybrid improved observed mean nDCG over BM25; reranking improved it further at **67.95 times** hybrid's mean latency. Dense had slightly higher Recall/Precision than hybrid, so hybrid did not dominate every metric.

[Exact measured report](experiments/reports/phase4_nfcorpus.json), [complete results and uncertainty](docs/experiment_results.md), and five figures/raw artifacts are linked from the external-validation docs. Different query distributions, judgment density/grades and timing conditions prevent treating cross-dataset score differences as controlled causal effects. Two broadly biomedical datasets do not establish universal generalization.

Final verification: **112 tests passed**; Ruff lint/formatting and dependency checks passed. Replays verified Phase 2 **1,200** rows, Phase 3 **300** rows and Phase 4 **1,615** rows. No API, dashboard, Docker, deployment, fine-tuning, rewriting or new adaptive policy was added.


## Phase 5: local FastAPI service

The API calls the existing retrieval implementations through lazy per-process resources. It supports `bm25`, `dense`, `hybrid`, `reranked` and the exact frozen `router`; prepared local models/indexes are reused. Health requests load no models. Searches have bounded serialized execution, strict input/body limits, request IDs, safe errors and JSON logs. Experiment CLI behavior and artifacts are preserved.

From the repository root, reuse the existing prepared SciFact corpus, models, index and frozen capsule:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[api,neural,dev]' --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m app.api
```

The service listens at `http://127.0.0.1:8000`; Swagger `/docs`, ReDoc `/redoc`, schema `/openapi.json`. In a second PowerShell terminal:

```powershell
$base = 'http://127.0.0.1:8000'
Invoke-RestMethod "$base/health"
Invoke-RestMethod "$base/methods"
Invoke-RestMethod "$base/models"
$body = @{ query = 'What evidence links vitamin D deficiency to disease?'; method = 'router'; top_k = 10 } | ConvertTo-Json
Invoke-RestMethod "$base/search" -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 10
$body = @{ query = 'What evidence links vitamin D deficiency to disease?'; methods = @('bm25','dense','hybrid','reranked','router'); top_k = 10 } | ConvertTo-Json
Invoke-RestMethod "$base/compare" -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 12
```

`top_k` is a strict integer 1..20 (default 10); nonblank query <=1,024 characters; body <=16KiB; compare <=5 unique methods. Compare returns rank overlap/order, not relevance metrics. Scores have method-specific meanings. Routing exposes its selected strategy, explanation/features and frozen policy ID. Initialization, complete-pipeline retrieval, rerank component and queue durations are distinguished. First-use costs are shared; the server never downloads resources or builds dense indexes through HTTP.

[Service guide](docs/api_service.md) gives exact dataset/model preparation, synthetic-only startup, environment settings, endpoint contracts/examples, errors, timing conventions, shutdown, installation recovery, tests and limitations. `.env.example` is reference only; set PowerShell `$env:RANKFORGE_...` variables explicitly. Use one worker and loopback; stop with Ctrl+C. This is a locally validated research service, not a production deployment.

Explicit real-server verification (requires the already prepared local SciFact resources):

```powershell
.\.venv\Scripts\python.exe scripts/verify_api.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

The verifier creates unique Phase 5 artifacts and reports, checks preserved query-1241 rankings, frozen policy, caching, safe errors and old artifact hashes. Ordinary tests use small authored fixtures/model doubles, with no download. Measured HTTP observations and final regression/replay outcomes are recorded in [experiment results](docs/experiment_results.md).


Phase 5 final checks: **159 tests passed**, Ruff lint/formatting (59 files) and dependency checks passed; Phase 2/3/4 replay verified 1,200/300/1,615 rows. All five methods matched saved top-10 rankings for public SciFact query 1241; frozen policy and single model initialization were verified. Three warm requests per method gave mean HTTP times BM25 **14.996ms**, dense **17.307ms**, hybrid **34.021ms**, reranked **1,328.148ms**, router **34.187ms**. These single-query observations are not a workload benchmark. [Exact measured report](experiments/reports/phase5-20261031T083003Z-c4676089f670.json) and the service guide record initialization costs, protected artifact hashes and limitations. Next command: `.\.venv\Scripts\python.exe -m app.api`.


## Phase 6: API reliability and resource validation

The [reliability guide](docs/api_reliability.md) documents a seeded mixed-query HTTP harness, per-concurrency/per-method latency and failure accounting, native process-memory sampling, direct/API correctness checks, and client timeout/cancellation recovery probes. Lifecycle counters and active/queued/stopping status are available through health/model responses. Shutdown now rejects new/queued work while allowing an active request to finish safely.

Reuse existing prepared resources and run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.api_load --requests 100 --clients 1,2,4,8,16 --timeout 30 --initialization-timeout 900 --warmup 10 --seed 42
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

Requests are counted per concurrency level; no retries hide 429s/timeouts. Eight public/authored queries include repetitions, keywords, questions, technical numbers and punctuation. The benchmark reads no qrels. It creates unique Phase 6 artifacts, raw request records and measured latency/memory plots. It runs its own loopback server and needs no additional runtime package. Client timeouts/cancellation do not interrupt running CPU inference; the lock remains owned until it finishes. See the [results](docs/experiment_results.md) and [decision log](docs/decisions.md) for actual tested levels, measurements, preserved failed runs and limitations.


Phase 6 measured final run: [recorded results](docs/experiment_results.md#phase-6-mixed-query-api-reliability-2026-11-09). Tested 1/2/4/8/16 clients with 100 attempts each: **280 successful, 220 HTTP429, zero workload timeouts**. All **360 correctness checks** passed; models/indexes initialized once; client cancellation/timeout recovery and active graceful shutdown verified. Working set grew from **61MiB** before loading to **1,192MiB** after work, with **1,234MiB** sampled overall peak; longer profiling is needed before a stability claim. Final **173 tests passed**, lint/format/dependencies passed, Phase2/3/4 replays passed, and 181 audit-protected prior files were unchanged. Four incomplete/invalid prototype runs remain documented; false launcher-memory measurements were explicitly invalidated. Increasing clients chiefly increased rejection; successful throughput is not a production capacity guarantee.


## Phase 6.1: sustained soaks and memory diagnosis

[Soak guide](docs/soak_testing.md) and [recorded results](docs/experiment_results.md#phase-61-sustained-memory-and-reliability-investigation-2026-11-09) document a completed same-process campaign: **15min C1,15min C2,5min C4 stress**, each followed by5min idle. Eight fixed public/authored queries, five equally weighted methods, explicit configurable memory/reliability budgets, native/Python allocation diagnostics, no forced GC or speculative memory optimization.

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.api_soak --duration 900 --clients 1,2 --stress-duration 300 --idle 300 --warmup 60 --sample-interval 1 --timeout 30 --seed 42 --window 120 --rss-budget-mib 2048 --growth-budget 2 --rejection-budget 0.05 --stress-rejection-budget 1 --p95-budget-ms 5000
```

C1:1653 successes/no rejection; C2:1307 successes/279 HTTP429; stress309 successes/21569 HTTP429. No workload timeouts/cancellations/other errors. **3621 correctness checks passed**, models/indexes initialized once, graceful shutdown verified. **185 tests passed**; lint/format67files, dependencies and Phase2/3/4 replays passed; 288prior files unchanged.

Memory sampled peak **1081.695MiB**; no confirmed leak, but **budgets did not all pass**: C1 late growth4.825MiB/min exceeded2; C2 rejection17.591% exceeded5%; stress p955754ms exceeded5000. Actual idle RSS changed0/0/-9.680MiB. Small request-object retention was investigated with a separate real-HTTP probe and disappeared during ordinary traffic without forced collection. Containerization readiness under the stated budgets remains unestablished. Failed/partial harness runs are preserved and explained in [results](docs/experiment_results.md); current API/retrieval behavior is unchanged.


Phase6.2 native profiling/repeatability: see [Windows commands, methodology and limitations](docs/native_memory_profiling.md). Run `.\.venv\Scripts\python.exe -m app.evaluation.native_repeat` for the full fresh-process campaign, then aggregate its printed directory with `.\.venv\Scripts\python.exe -m app.evaluation.native_analysis PATH`. This is a long diagnostic run, not a deployment command.
