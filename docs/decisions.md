# RankForge implementation progress

- [x] Audit: empty workspace, no Git repository, no ancestor AGENTS.md found. Python default 3.12.5; py -3.11 is 3.11.5.
- [x] Scope: Phase 0/1 only. No API, dashboard, Docker, dense models or reranking.
- [x] Create Python 3.11 environment and constrained dependencies.
- [x] Implement validated BEIR-style data, synthetic fixture, typed BM25, metrics.
- [x] Test and run benchmark; record real outputs and commands.

Decisions: use rank-bm25 Okapi, standard-library dataclasses and explicit validation; pytest for tests. Public SciFact download deferred; verify format/license and document source first. Synthetic results never represent public dataset performance.


## Completed implementation and verification

- [x] Python 3.11.5 local environment; pinned rank-bm25 0.2.2, NumPy 2.2.6, pytest 8.4.2, Ruff 0.14.2.
- [x] Validated schemas and BEIR-style loader; explicit split selection and judgment joins.
- [x] Clearly labeled synthetic fixture: six documents, four queries, four positive judgments.
- [x] Typed BM25 retrieval, deterministic ties, shared Unicode/identifier tokenization.
- [x] Recall, Precision, MRR and graded nDCG with hand-calculated test cases.
- [x] Integration test verifies saved outputs, query joins, finite metrics and unique run directories.
- [x] CLI benchmark completed; inspected summary and metadata. Exact numbers in experiment_results.md.
- [x] Final pytest: 29 passed. Ruff lint and format verification run.

Reproduced issue: first pytest run returned 1 failed / 28 passed because the PowerShell pipeline replaced non-ASCII test literals with question marks. Fixed by using Unicode escapes, preserving the Unicode assertions. Subsequent pytest returned 29 passed. Format check then requested reformatting the modified test; applied formatting and reran checks.

Environment audit command `git status --short` failed because this workspace has no .git directory; no Git history or existing implementation was present. No repository was initialized or published. No ancestor AGENTS.md found in inspected ancestor directories. Runtime dependencies were missing initially and installed into .venv. No unresolved Phase 1 test failures.

Public SciFact format/license sources are documented in data/README.md. At the end of the initial Phase 1 milestone, public download and evaluation were deferred; the completed public-data milestone below supersedes that status. Smoke results remain separate. Phase 2 onward is deliberately unimplemented. No API, dashboard, Docker, models, deployment, paid services or external publication.

Next commands from the repository root (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m app.evaluation.benchmark --dataset data/smoke --split test --k 10
```

For a new environment, follow README.md installation instructions. Public-data research requires provenance capture and a frozen development/held-out protocol before reporting results. The initial Phase 1 milestone did not provide a download command; the public-data milestone below adds one.

## Public-data milestone plan (2026-10-25)

- [x] Inspect existing architecture and reproduce baseline: 29 tests passed.
- [x] Verify BEIR SciFact archive source, published MD5, dataset-card CC-BY-SA-4.0 license, and license terms. References in data/README.md.
- [x] Add checksum-verified staged download/import with provenance and no dataset overwrites.
- [x] Add deterministic query subsets while retaining the complete corpus and original qrels.
- [x] Add offline parser/join/subset/download workflow tests and run all checks.
- [x] Run 25-query and complete test-split BM25 evaluations; preserve separate output identities and document measured results.

Freeze before evaluation: existing tokenizer, BM25 k1=1.5, b=0.75, epsilon=0.25, k=10; no tuning on test. Subsets use SHA256(seed + ':' + query_id) ordering, seed 42 by default. Full run uses all queries in the selected official split. This milestone does not add retrieval models or services.

## Public-data milestone outcomes

- Reused existing Dataset, loader, BM25 and metrics; no retrieval algorithm or metric formula changed. Loader errors now explicitly reject non-object JSONL and malformed TSV widths.
- Added `app.data.download` (stdlib only) and `app.data.selection`; benchmark accepts `--max-queries` and `--seed`. Metadata now records subset/evaluation conventions and code hashes.
- Added 17 offline tests; complete pytest run: **46 passed**. Final checks: pytest **46 passed in 0.56s**; Ruff lint **All checks passed**; Ruff format **15 files already formatted**. Offline dataset reuse also succeeded.
- Direct Python HTTPS download failed certificate verification. Windows `Invoke-WebRequest` succeeded; importer verified official MD5. Data is local and ignored, with CC-BY-SA-4.0 attribution and immutable archive identity recorded.
- Ran subset `20261025T174241Z-b4b56da62891` and full test `20261025T174246Z-5ea5ddf8cefd`. Exact measured values and raw artifact paths are in experiment_results.md and experiments/reports/scifact_bm25.json. Original synthetic run remains intact.
- Replayed all full-run per-query metrics from rankings/qrels; macro summary matched. Subset rankings match full run for all 25 queries. BEIR train and test query IDs are disjoint.
- No unimplemented later phases were started. Public-data evaluation is complete; Python's direct HTTPS trust issue remains environment-specific, with a verified Windows download/import route documented.

Next commands (from repository root, Windows PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m app.data.download --archive data/public/archives/scifact.zip
.\.venv\Scripts\python.exe -m app.evaluation.benchmark --dataset data/public/scifact --split test --max-queries 25 --seed 42 --k 10 --output experiments/results/scifact
.\.venv\Scripts\python.exe -m app.evaluation.benchmark --dataset data/public/scifact --split test --k 10 --output experiments/results/scifact
```

For a missing archive or network retry, use README.md's verified PowerShell download block, then `--archive`. Future tuning must use the BEIR train split; test was evaluated only with pre-existing frozen settings. Do not interpret this milestone's test subset as a held-out validation set.

## Phase 2 plan and frozen comparison settings

- [x] Inspected existing implementation, prior SciFact artifacts and model interfaces. Baseline pytest: 46 passed in 0.49s.
- [x] Confirmed original BEIR test configuration: 5,183 documents, 300 queries, 339 judgments; k=10, unchanged tokenizer/metrics and BM25 k1=1.5, b=0.75, epsilon=0.25. Existing runs will be preserved.
- [x] Dense cosine retriever, normalized persistent embeddings, strict cache validation and offline tests.
- [x] Equal-weight RRF (constant 60, depth 50 per source) and candidate-only cross-encoder (pool 20), offline tests.
- [x] Fair comparison runner, model/index timing, warmup, candidate/failure artifacts and replay checks.
- [x] Download pinned model snapshots; run a 25-query test subset, then full test if CPU runtime permits.
- [x] Complete checks and update measured results, limitations and exact commands.

Frozen before scoring: all-MiniLM-L6-v2 dense model (256-token maximum), ms-marco-MiniLM-L6-v2 cross-encoder (512-token query/document maximum), CPU, 4 torch threads, batch size 16; RRF constant 60; source depth 50; cross-encoder pool 20. No parameter tuning on test. Dense/cross-encoder input is original title + space + text (no lexical tokenizer). Warmup uses authored non-evaluation text; every method receives one unmeasured warmup search. Timed pipelines execute independently, recomputing their dependencies so hybrid and rerank latency include retrieval cost. Results go under experiments/results/phase2 with unique IDs.

Use optional neural dependencies; ordinary tests remain runnable without downloading models. New local model/index artifacts will be ignored. Model cards (Hugging Face) list Apache-2.0; no fine-tuning or external inference service is used. If downloads fail, record the error and retain working lexical commands.

Dense focused verification: 17 passed in 0.21s; Ruff passed. Model download was first invoked before dependency installation finished and failed with missing huggingface_hub; retry follows installation.

Phase 2 focused checks: RRF/reranker 17 passed; combined retrieval 34 passed. Complete suite after comparison integration: 87 passed; after snapshot/error/percentile coverage: 92 passed. Model downloads succeeded after dependency installation; dense index built for 5,183 documents with 384 dimensions. Public comparison is in progress; no scores claimed yet.

Initial real-model public subset completed: phase2-20261025T175857Z-08cbfb3c0287 (25 queries). Mixed quality: hybrid recall 0.72 vs reranked 0.68, with reranked mean latency about 880ms. No tuning or configuration changes; proceed with full 300-query run. Replay now also verifies candidate prefixes, disagreement analysis and summary CSV.

## Phase 2 completed outcomes and next commands

- Implemented dense cosine/cache validation, equal-weight RRF and bounded cross-encoder reranking behind the existing Retriever/SearchResult contract. Existing BM25, datasets, loader, tokenizer and metric formulas were not changed in this phase.
- Added model snapshot download/import provenance; local-only neural inference with remote code disabled. Optional dependency installation succeeded. First model download failed only because dependency installation was still running; retry succeeded using system HTTPS trust.
- Both model snapshots are pinned by full commit and file hashes, documented in phase2.md. SciFact dense index built successfully: 5,183 x 384 float32 vectors; original build 95.411672s. Full/subset runs used a validated cache hit; setup and model loading timed separately.
- Completed subset phase2-20261025T175857Z-08cbfb3c0287, full phase2-20261025T180228Z-bdac57e0c104, and separate real-model smoke phase2-20261025T180750Z-edfa75e06081. Replay verified 100, 1,200 and 16 rows respectively. Full and subset BM25 rankings/scores exactly match the preserved Phase 1 runs. No benchmark failed or existing output was overwritten.
- Full run observed nDCG: BM25 0.633399, dense 0.645082, hybrid 0.676569, reranked 0.700349. Reranking improved 65/degraded 53/preserved 182 query-level nDCG values. Mean hybrid/reranked latency 30.161/938.427ms. These are measured descriptive results, not a significance or universal-superiority claim. Exact outputs and genuine examples are in experiment_results.md and experiments/reports/phase2_scifact_full.json.
- Final checks: **93 passed in 1.44s**, Ruff lint passed, Ruff format reports **26 files already formatted**, pip check found no broken requirements. Ordinary tests require no network/models/public-data downloads. Full failure/candidate/CSV/JSON replay is implemented and verified.
- Remaining limitations: truncated inputs, sparse qrels, domain transfer, exposed test set, single-pass warmed timings, no peak-memory or throughput study. No routing, API, dashboard, Docker, training or paid service was added. The broader Phase 3 Failure Lab remains future work.

Actual commands executed (PowerShell, repository root; also documented in README):

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[neural,dev]'
.\.venv\Scripts\python.exe -m app.data.models
.\.venv\Scripts\python.exe -m pytest tests/test_dense.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_hybrid_reranker.py -q
.\.venv\Scripts\python.exe -m app.retrieval.dense --dataset data/public/scifact --cache indexes/scifact-minilm.npz --batch-size 16 --threads 4
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --max-queries 25 --seed 42 --require-cache --output experiments/results/phase2
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --require-cache --output experiments/results/phase2
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --replay experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104
.\.venv\Scripts\python.exe -m app.evaluation.report --run experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104 --output experiments/reports/phase2_scifact_full.json
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/smoke --cache indexes/smoke-minilm.npz --candidate-depth 5 --rerank-candidates 3 --k 3 --output experiments/results/phase2-smoke
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

Exact next step: replay the full completed run using the command above, then review the recorded gains/regressions before authorizing any new experimental scope. To rerun the same comparison, use the full comparison command above; a new run directory is always created. To recreate model snapshots, use README's explicit revision flags. Rerunning report export requires a new output filename; it refuses overwrite. No additional implementation phase has been started.

## Phase 3 plan (2026-10-31)

- [x] Inspect existing modules/artifacts/docs; baseline 93 tests passed in 1.45s; Phase 2 replay verified 1,200 rows.
- [x] Extract corpus/query features without judgments, explainable evidence-based multi-label taxonomy, all-query failure tables.
- [x] Persist SHA256(seed:query_id) exploratory 60/20/20 split (180/60/60 for 300 exposed BEIR test queries).
- [x] Propose a small fixed router rule family on development; select thresholds on validation only; persist frozen policy and tuning audit before held-out evaluation.
- [x] Fresh same-query CPU timing of all four baselines and live router on held-out IDs, including routing overhead; paired bootstrap intervals.
- [x] Save/replay required CSV/JSON artifacts, derive five analytical plots, update docs and run final checks.

Design before coding: source Phase 2 pipeline settings, metrics and qrels remain unchanged. Router consumes only query text/corpus lexical statistics (no dense inference or labels). Diagnostic dense/BM25 score/agreement features come from existing rankings and are not router inputs. Rule family: BM25 for sufficiently rare/covered terms; otherwise rerank short queries under a threshold; hybrid otherwise. Thresholds will be selected with explicit quality/latency utility and reranker budget on validation. Exposed test split remains exploratory, never described as untouched external validation. Historical failure analysis uses Phase 2 timings; final held-out comparison is freshly timed independently with model/index loading excluded. Frozen policy is persisted before accessing held-out outcomes for routing evaluation; no post-evaluation threshold updates. Sparse judgments and taxonomy hypotheses will be explicitly labeled. No API, dashboard, containers, deployment or fine-tuning.

Phase 3 implementation checkpoint: features/split/taxonomy/router and offline orchestration tests pass (10 passed in 0.90s); Ruff passes after removing one unused import. Plot dependency installation first failed in the network-restricted sandbox (setuptools index unavailable), then succeeded with approved network access. No model or dataset download was needed. Before held-out scoring, policy utility is fixed to nDCG minus 0.1 times historical latency in seconds, with a <=50% reranking constraint on development/validation; six development-shortlisted policies plus the no-rerank hybrid fallback reach validation. No thresholds will change after held-out scoring.

## Phase 3 completion (2026-10-31)

- Run: phase3-20261031T065803Z-72804b2af31a. Full historical Failure Lab: 300 queries x four methods; fresh live internal-held-out evaluation: 60 queries x five methods. No original Phase 1/2 artifact or metric/retrieval module was modified.
- Frozen validation policy selected the hybrid-only fallback. Retained negative result: no adaptive improvement demonstrated. Held-out route shares: 100% hybrid / 0% BM25 / 0% reranking. Router nDCG 0.639072 vs reranked 0.689126, mean latency 31.550ms vs 896.722ms; 96.48% mean savings but lower average quality. Paired exploratory intervals and all metrics are in the report; no threshold was updated after evaluation.
- Exact split IDs (180/60/60), development/validation-only outcome access, policy digest and event order were persisted and replayed. Live inference accepts no qrels/metrics; route explanations and selected independent-system rankings replay correctly. Prior Phase 2 integrity replay checks old labels mechanically and is explicitly not claimed to restore an untouched benchmark.
- All nine required artifacts and five analytical plots exist. Extra files contain frozen policy, tuning audit, live rankings/rows and hashes. Portable measured report: experiments/reports/phase3_scifact.json. Visual label-layout fix rendered figures again from the same saved data; separate render code hash recorded, no model/policy rerun.
- Final tests: **104 passed in 1.67s**; Ruff lint passed; formatting **35 files already formatted**; pip check passed. Phase 2 replay **1,200 rows**; Phase 3 replay **300 rows**. No remaining failed checks or experiment failures.
- Resolved: network-restricted plot dependency installation required approved access; one unused import failed lint and was removed; overlapping plot labels fixed after visual inspection. Matplotlib 3.10.7 is optional analysis dependency. No models/datasets were downloaded again.
- Limitations and negative findings are documented in ranking_failure_lab.md, query_aware_routing.md and experiment_results.md. Routing on a previously inspected benchmark is exploratory; the chosen family does not justify an adaptive deployment claim. No API/dashboard/Docker/cloud/authentication/fine-tuning/LLM rewriting was started.

Actual principal commands executed (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[analysis]'
.\.venv\Scripts\python.exe -m pytest tests/test_phase3.py -q
.\.venv\Scripts\python.exe -m app.evaluation.phase3
.\.venv\Scripts\python.exe -m app.evaluation.phase3 --replay experiments/results/phase3/phase3-20261031T065803Z-72804b2af31a
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --replay experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104
```

Next exact step is the Phase 3 replay command above and review of the negative routing finding. A fresh `python -m app.evaluation.phase3` run creates a new directory, but must not be used for iterative held-out tuning. For future external validation, choose a new dataset and freeze the policy/protocol before examining its labels.

## Phase 4 plan and external-transfer registration (2026-10-31)

- [x] Audit retrieval/evaluation/download/cache/routing/docs and prior artifacts; 104 tests passed in 3.21s; Phase 3 replay verified 300 rows.
- [x] Freeze exact saved Phase 3 policy before NFCorpus label access; preserve canonical and byte hashes plus unchanged policy implementation hashes.
- [x] Integrate official checksum-pinned BEIR NFCorpus archive, preserving original ZIP and extracted files; inspect all splits and record observed counts/grades. Official test is preselected for evaluation, no tuning.
- [x] Add external comparison using unchanged metrics/retrievers/router, fresh separate dense index, independently timed five-system searches, actual candidate traces and shared failure taxonomy.
- [x] Run full NFCorpus test evaluation; save required artifacts, paired intervals, five data-derived plots, replay and backward compatibility checks.
- [x] Document measured transfer findings and limitations, including narrow biomedical domain shift and graded/sparser-vs-denser judgment effects.

Before external labels: saved Phase 3 canonical policy record hash is 31b0cef1d5969d2b79953be683050877038c34fabc4c560b4a1b8be8f90b67f8; parameters rare_min=1.01, rerank_max_tokens=0, coverage_min=0.8. It imports without behavioral modification. All model/pipeline settings remain inherited; only corpus/cache changes. New dataset is BEIR NFCorpus from official UKP source, published MD5 a89dba18a62ef92f7d323ec890a0d38d; BEIR dataset card lists CC-BY-SA-4.0. Original project website timed out; official BEIR listing/card and CC terms were verified. No labels or raw data were loaded before the transfer protocol freeze. No new dependencies or tuning are planned.

Phase 4 pre-evaluation checkpoint: frozen protocol created 2026-10-31T07:11:51.226254+00:00; NFCorpus prepared at 07:12:48.494563+00:00 after checksum verification. The sandbox network request was refused (WinError 10061), approved retry with system certificate verification succeeded. Raw ZIP retained at data/public/archives/nfcorpus.zip, SHA256 efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b. Observed 3,633 documents, 3,237 total query IDs; train/dev/test 2,590/324/323 queries and 110,575/11,385/12,334 judgments. Test grades: 11,758 grade-1 and 576 grade-2. Splits disjoint; no unaccounted query IDs, no filtering. Official test is selected per frozen registration. Dense index built separately for 3,633 x 384 vectors; no SciFact cache reuse. Focused tests 7 passed; full suite 111 passed in 5.29s; Ruff lint passed and 40 files formatted. Two unused imports caught during implementation were removed. No external policy tuning; proceed with full 323-query run.

## Phase 4 completed outcomes (2026-10-31)

- Full external run: phase4-20261031T071828Z-46027d7d4330. All 323 NFCorpus test queries evaluated by five independent systems; 3,633 documents and 12,334 original graded judgments. No corpus/query filtering, label fabrication or policy/model retuning.
- Exact Phase 3 frozen record/byte/implementation hashes verified before preparation and after evaluation. Canonical policy ID remains 31b0cef1d5969d2b79953be683050877038c34fabc4c560b4a1b8be8f90b67f8. Routes: 100% hybrid, 0% reranker invocation; identical hybrid/router ranks/scores and metrics. No new adaptive policy was introduced.
- Measured nDCG: BM25 0.294681; dense 0.317272; hybrid/router 0.324066; reranked 0.351809. Hybrid-minus-BM25 +0.029385 and reranked-minus-hybrid +0.027743; paired query-bootstrap intervals exclude zero on this collection/pass. Dense-minus-BM25 interval spans zero. Dense Recall/Precision exceed hybrid slightly, so no all-metric dominance claim.
- Reranking mean latency 1,896.101ms versus hybrid 27.903ms (~67.95x); frozen router 26.119ms, saving 98.62% versus reranking while losing 0.027743 nDCG. Independent hybrid/router timing differences are noise/overhead, not algorithmic speedup. Reranking improved 129, degraded 75 and preserved 119 query-level nDCG values.
- Failure taxonomy retained observed/inferred distinction. Candidate miss indicator on 307 queries is any positive missing; 74 have no positive candidate. Judgment density/grades and task differ substantially from SciFact; descriptive score/failure-rate changes do not establish causal degradation or universal generalization.
- All nine required artifacts and five figures exist; portable exact report experiments/reports/phase4_nfcorpus.json. Figures inspected. All current Python implementation hashes match the measured run. Original retrieval/metrics/router and Phase 1-3 results were not rewritten.
- Final checks: **112 passed in 7.13s**, Ruff lint passed, formatting **40 files already formatted**, pip check passed. Replays: Phase 2 1,200 rows; Phase 3 300 rows; Phase 4 1,615 rows. A deliberate policy-mutation fixture confirms incomplete runs are rejected; it is not a public benchmark failure. Full external benchmark completed with no inference failures.
- Resolved setup issues: restricted sandbox download WinError 10061, approved verified HTTPS retry succeeded; original project website timed out, BEIR card/checksum/license terms and primary publication were accessible; two unused imports removed after lint. No remaining critical blocker. No API/dashboard/Docker/cloud/authentication/fine-tuning/LLM rewriting was added.

Commands actually executed (PowerShell, repository root):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m app.evaluation.phase3 --replay experiments/results/phase3/phase3-20261031T065803Z-72804b2af31a
.\.venv\Scripts\python.exe -m app.evaluation.transfer_policy
.\.venv\Scripts\python.exe -m app.data.nfcorpus
.\.venv\Scripts\python.exe -m app.retrieval.dense --dataset data/public/nfcorpus --cache indexes/nfcorpus-minilm.npz --batch-size 16 --threads 4
.\.venv\Scripts\python.exe -m pytest tests/test_phase4.py -q
.\.venv\Scripts\python.exe -m app.evaluation.phase4
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m app.evaluation.comparison --dataset data/public/scifact --replay experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104
.\.venv\Scripts\python.exe -m app.evaluation.phase4 --replay experiments/results/phase4/phase4-20261031T071828Z-46027d7d4330
```

Next exact command: replay the completed Phase 4 run using the last command above. A fresh phase4 run creates a separate directory, but external results must not be used to alter this frozen transfer policy. Recommended next scope: engineering hardening/product API with explicit quality/latency requirements; repeated timing and broader-domain validation remain worthwhile separate analyses.

## Phase 5 plan (2026-10-31)

- [x] Audit docs/interfaces, retrieval/cache/model lifecycle, frozen router, persistence and replay. Baseline 112 passed in 5.26s; Phase 2/3/4 replay verified 1,200/300/1,615 rows.
- [x] Centralized service configuration and lazy shared resource lifecycle; use existing retrievers without algorithm changes. Corpus-only API loading, no qrels or benchmark-writing endpoint.
- [x] Typed /health, /methods, /models, /search, /compare endpoints, safe errors, request IDs, JSON logs and size/concurrency bounds.
- [x] Offline tests for endpoints, schemas, failure paths, comparisons, frozen routing, concurrency and no-download health checks.
- [x] Install/verify optional API dependencies; start a local server, exercise every endpoint with real prepared resources, check cold/warm reuse and unchanged rankings.
- [x] Complete checks/replays, verify all earlier artifact hashes unchanged and document exact commands/measurements/limitations.

Design before coding: lazy resources per process, no model initialization on module import or health requests. Read-only prepared dense cache required; model adapters stay local-only. All CPU searches serialized per manager with bounded admission/queue waiting; health snapshots use a separate short-held status lock. Configuration is server-side environment only, no request-controlled paths/models. top_k 1..20, query <=1,024 characters, body <=16KiB default, compare <=5 unique methods. CORS disabled by default, exact origins only if configured. Use existing frozen transfer capsule to preserve Phase 3 behavior; do not load the cross-encoder for a hybrid-only router. Model/index initialization measured separately, no automatic artifact modification or downloads. Local validation only, not a production-deployment claim.

Phase 5 checkpoint: lazy resource/services and all five required HTTP endpoints implemented. Focused offline API suite: 39 passed in 2.62s. One upstream TestClient deprecation warning appeared with unbounded AnyIO 4.15.1; constrain API AnyIO to 4.11.0 (compatible release for the pinned Starlette) and verify instead of suppressing the warning. Initial API dependency install failed inside restricted package-index access; approved retry succeeded. BM25 synthetic TestClient request and OpenAPI returned 200. No retrieval/metric/router source was changed.


Phase 5 completed: 47 authored/offline API/service tests added; final complete suite **159 passed in 7.82s**, no warnings. Ruff lint passed, formatting **59 files already formatted**, pip check found no broken requirements. Phase 2/3/4 replays verified **1,200 / 300 / 1,615** rows after implementation. All old recorded source/retrieval/policy hashes remain valid. Current service source hashes match final real-HTTP verification.

Real Windows process/loopback run: **phase5-20261031T083003Z-c4676089f670**, complete. Every endpoint, OpenAPI/docs HTML, all five search methods and comparison succeeded; safe 422/413/404/405 cases verified. SciFact query 1241 top-10 IDs/ranks matched preserved Phase 2 for every method (router equals hybrid); scores matched within 1e-5. Canonical policy identity unchanged. Encoder/scorer each initialized once; hybrid-only router left scorer unloaded until explicitly requested. Full queries/canary absent from logs. All **134** previously existing protected experiment/config/model/index files in the saved manifest remained byte-identical (this includes the earlier Phase 5 report). First verification run phase5-20261031T082604Z-0f3d47060a77 also succeeded and is preserved; final run rechecks the completed schemas/OpenAPI examples and source manifest.

Actual resolved issues: one initial full-suite run had **1 failed, 155 passed** because FastAPI omitted a null routing field from the recorded OpenAPI example; making the nullable routing field optional restored schema/example compatibility. A focused run initially emitted an upstream Starlette TestClient/AnyIO deprecation warning; the tested AnyIO 4.11.0 pin removes it without suppressing warnings. Package-index DNS failed for the isolated-build retry; no-build-isolation then exposed missing bdist_wheel. Installing AnyIO 4.11.0, setuptools 84.2.1 and wheel 0.48.0 from explicit public PyPI succeeded, followed by a successful offline editable install. No unresolved check failure remains. Fresh-environment installation was not tested.

No model/data download, dense-index build, scientific re-evaluation, adaptive-policy change, dashboard, Docker or deployment was performed. HTTP timings are separate engineering observations, not new quality metrics. The verification helper terminates only its own Windows child; graceful application lifespan cleanup is covered by TestClient. Final verification server has stopped. Exact next command: `.\.venv\Scripts\python.exe -m app.api`; see docs/api_service.md for all setup/call/replay commands. Next recommended milestone: mixed-query concurrency, timeout/cancellation and memory-budget validation before any product dashboard/deployment.


## Phase 6 plan and audit (2026-11-09)

- [x] Read Phase 6 specification, API/service/model/configuration/tests, service guide and Phase 5 measured report. Existing suite: 159 passed in 8.62s. Preserve prior artifact/model/index and retrieval/evaluation/data code hashes before edits.
- [x] Regression-test and repair shutdown admission race; expose cheap lifecycle attempt/success and active/queued counters without model loading.
- [x] Build seeded mixed public/authored query HTTP harness, configurable request count/concurrency/timeout/method weights/warmup/seed. Never read qrels to schedule requests.
- [x] Compare multiple-query direct retriever results against API before/after load, verify policy identity, no eager reranker and cache reuse. Measure child RSS/working set at lifecycle stages and sampled peaks.
- [x] Run real CPU mixed loads at 1/2/4/8/16 clients, plus cold initialization and timeout/cancellation recovery probes. Add deterministic fault tests for slow/throwing inference, cancellation and active shutdown.
- [x] Save all seven required outputs, raw requests/resources, workload/config/source/environment metadata and figures; retain every failure/429/timeout.
- [x] Full tests/API integration, Ruff, dependency checks, Phase 2/3/4 replay, previous-artifact hash verification and documentation with exact commands and limitations.

Audit: shared model objects, dense arrays and corpus/index/router references are mutable Python objects but API inference is serialized by the work lock. Initialization uses a separate reentrant lock; status snapshots use a short status lock. There is no cache eviction or HTTP mutation endpoint. CPU work cannot be safely killed in a Python thread on client timeout; retain serialization until it finishes. Identified defect: close() sets closed only after acquiring the busy work lock, so new/queued work can enter while shutdown is waiting. Fix admission immediately while allowing already active retrieval to finish. Keep retrieval/routing algorithms and Phase 1-5 artifacts untouched. Use native Windows process-memory counters (no additional runtime dependency); report sampled working set rather than pretending this is exact allocated/peak memory. Real server controlled shutdown is benchmark-process private control-file only, never an API control endpoint.


Phase 6 completed run: phase6-20261109T153436Z-0e8e7412820d. All 1/2/4/8/16-client levels actually ran, 100 requests each, five equal-weight methods/eight public-authored queries, seed42/warmup10/timeout30. 280/500 successful, 220 HTTP429, no measured-workload timeouts. Cold test 1 success/3 HTTP429; encoder/index each initialized once. All 360 direct/before/during/after correctness checks passed, frozen policy unchanged, no eager scorer for router. Actual CPU timeout/task-cancel probes retained active lock until completion and recovered; real active graceful shutdown returned matching HTTP200 and exit0.

Resource validation fixed Windows venv-redirector PID selection: sample actual serving PID8640, not launcher12068. Working set before loading61.074MiB, after all initialization711.367MiB, after warmup872.348MiB, after measured requests1191.965MiB; load sampled peak1170.973MiB, overall1234.086MiB. Growth remains unclassified, no memory-leak/stability guarantee. Required CSV/JSON and measured PNGs exist and figures inspected. Native counters require no added dependency; WMI total-system-RAM query denied and omitted.

Four incomplete/invalid prior Phase6 runs retained (150740Z-fc24027bd07c,151804Z-00f856221040,152612Z-808155d5ca11,152911Z-02e5fae500b8). First three used stdin-control readers that interfered with native imports; first hit cold deadline, next two were deliberately stopped. File-based benchmark-only control resolved it. Fourth finished timing/correctness but watched launcher memory and was explicitly invalidated as failed after review; no false ~4MiB server-memory claim. Diagnostic traces and requests preserved. One intermediate suite165pass/6fixture-errors fixed; final173passed9.49s, Ruff lint/format62files and dependencies passed. Final replays1200/300/1615, all181original audit hashes unchanged; current measured source hashes match. Docs record timing/accepted-mix/background-activity limits. No unresolved correctness check, but 429 admission limits, CPU non-interruptibility and growing working set remain operational limitations. Exact next benchmark command is in docs/api_reliability.md; next work should be fixed-concurrency soak/profiling with a stated memory/error budget.


## Phase 6.1 investigation plan (2026-11-09)

Audit completed before implementation: existing harness/sampling/PID selection, reliability guide, lifecycle/queue/cancellation, dense temporary arrays/model adapters and Phase6 complete/failed reports inspected. Preserve all previous artifacts and application implementations before edits. Existing test baseline recorded separately. Candidate explanations: expected weights/index memory; transient embeddings/scores/pair tensors and serialization; native allocator/workspace retention; Python request/response or background-task retention; instrumentation retention. None establishes a leak. Phase6 tracemalloc absence and short mixed-concurrency duration cannot separate these.

- [x] Add duration-based fixed-concurrency harness reusing validated PID, workload, aggregate, drain and atomic private control mechanisms; preserve the existing API/retrievers.
- [x] Benchmark-only Python/native diagnostics, tensor/index byte counts, object/thread/task counts, available memory where supported; no forced GC/cache clearing or speculative optimization.
- [x] Pre-register configurable research budgets: peak RSS 2048MiB, late RSS slope 2MiB/min, normal rejection <=5%, stress rejection <=100% (intentional overload), zero timeout/cancellation/other errors, successful p95 <=5000ms. These are local investigation targets, not universal production thresholds. Classify insufficient windows as indeterminate; report violations without hiding completed runs.
- [x] Short smoke, then recommended 900s C1 +300s idle, 900s C2 +300s idle in same serving process, optional stress300s C4 +300s idle; warmup60s, sampling1s, all five equally weighted methods, seed42, original eight query pool. Same process enables repeated-soak recovery checks. Preserve partial runs if interrupted.
- [x] Save eight required artifacts, four figures, diagnostics, correctness/lifecycle, budgets, software/hardware/source metadata. Run full tests/Ruff/dependency checks and Phase2/3/4 replays; verify all old hashes; document measured stability and limits before containerization recommendation.


Phase6.1 completed campaign phase6_1-20261109T162840Z-f2dea9f54ecb: actual900s C1,900s C2,300s C4 stress with300s idle each, same servingPID20388 (launcher21548),60s warmup,1s sampling,trace-on after model initialization. C1 1653/1653 successes; C2 1307/1586 with279 HTTP429; stress309/21878 with21569 HTTP429. No other HTTP errors/timeouts/cancellations. 3621 correctness checks passed, model/index attempts/counts all1, same policy, final graceful shutdown exit0.

Budgets deliberately not relabelled: C1 late RSS4.825MiB/min failed2 target; C2 rejected17.591% failed5% normal target; stress p955754.006ms failed5000ms. Completed run != budget pass. RSS beforeinit60.926MiB/afterinit626.754/afterwarm924.578, overall sampledpeak1081.695MiB, OSpeak1085.879. Late slopes C1+4.825/C2-3.628/stress-0.405. Idle itself changed0/0/-9.680MiB; original workload-end-to-idle-end delta includes intervening diagnostic allocations and is explicitly clarified. Native attribution incomplete; no confirmed leak and no product memory optimization. Stable tensor6/model bytes/index bytes; trace current0.423?1.124MiB, totaltrackedobjects declined389996?387630. Selected request objects16/20 persisted after rejection; isolated authored HTTP probe3after50rejects/idle?0after100normalrequests supports collectibility without forcedGC, not universal leak-free proof.

Retained failed smoke155639 descriptor issue, successful smoke155820, and partial long160233 (900sC1+300sidle, no post-idle completeness due controller PermissionError). Diagnostic descriptor handling/read retries/atomic reply fix each supported by regression tests; finalcampaign completed. Original API/retrievers untouched; all288oldhashes preserved. Final185tests18.66s, Ruff67files/dependenciespassed, replays1200/300/1615. Eight required artifacts/four figures inspected; exact commands docs/soak_testing.md. No budget relaxation or Docker implementation. Container readiness under stated budgets not established; next native profiling/admission-target work before deployment claims.


## Phase 6.2 profiling plan (2026-11-10)

Audit: Phase6.1 tools/report, failed/partial runs, memory/PID selection, lifecycle/queues/inference, native NumPy/OpenBLAS/PyTorch usage and soak guide inspected. Prior Phase6.1 baseline:185tests. Verified Phase6.2 full rerun:191passed29.85s (six new native-analysis tests). NumPy threadpoolctl reports OpenBLAS0.3.29 pthreads20, while adapters configure torch threads4; this is observation, not yet attribution. Windows VirtualQueryEx and Toolhelp thread enumeration documentation checked; copy-on-write mapped/image pages remain classified mapped/image, so region sums cannot be equated to PrivateUsage or heap-owner attribution.

- [x] Add opt-in benchmark-only native diagnostics (working set/private commit, virtual reserved/committed type totals, native threads, loaded DLLs, torch/BLAS settings observed in actual inference threads). Production service/retrievers untouched. Retain original api_soak source before a backwards-compatible optional profiling flag.
- [x] Short profiling smoke; independent fresh-process runs C1 twice/C2 twice at900s each plus300s C4 stress,60s warmup/300s idle/1s sampling/30s timeout/seed42, identical eight-query/equal-five-method cycle. Existing budgets2048MiB RSS/2MiB per minute late growth/5% normal rejection/100% intentional stress rejection/5000ms successfulp95/zero other errors unchanged. No limit tuning for favorable scores.
- [x] Isolated one-variable native controls: OpenBLAS1vs20 before/after NumPy import and identical normalized matrix-vector work; hybrid versus reranked repeated fixed-query inference with all resources initialized, plus representative CPU operator-memory profiler where supported. No forced GC or allocator release.
- [x] Aggregate required eight artifacts and five plots; classify budgets passed/failed/inconclusive and two-run variability without claiming confidence intervals. Preserve failures and all original artifacts; record native attribution limits and Docker/single-/multi-user/production evidence separately.
- [x] Full tests, Ruff/dependencies, correctness/shutdown, prior replays and hashes; exact commands, environment and readiness conclusions.

Phase6.2 progress: smoke completed (7/7correct measured requests; p95failed; memory-duration inconclusive). First independent900s C1 completed1492/1492success, p952779.231ms, late RSS+8.440MiB/min (growthfailure). First C2 completed1924attempts/1600success/324rejected, p952703.566ms, late RSS+4.037 (rejection/growthfailures); no timeouts/other errors. Both exited0. A review of the completed CSV found the opt-in native sampler was patched on the canonical module instead of the running __main__ module. RSS/private measurements remain1s and native snapshots/configuration/lifecycles are valid; continuous native fields are missing. Preserve comparability of all five runs; export native checkpoints honestly, fix wiring and regression-test/short-smoke it after the campaign, without claiming a continuous native trace for prior runs.


Phase6.2 completed: five independent full-duration runs/idle periods, all exited0; 30143measured attempts,6505success,23638intentional/admission rejections, zero request timeouts/cancellations/other errors.7416correctness checks passed. All four normal runs failed growth; both C2also failed rejection; stress failed successfulp95. Leak attribution remains Inconclusive. BLAS import and full-pipeline controls identify large private-commit contributions; fixed-query loops/profiled allocations do not prove leak absence. No service configuration/algorithm/policy/memory optimization changed.

The initial inference control had a30014.849sinitialization gap after all serving runs had finished. Preserve/exclude that attempt; all three conditions rerun clean in fresh processes. The native sampler __main__ wiring fix passed its regression and real smoke:97/97native observations,15/15steady successes,99correctness,exit0,no sampler errors. Its198.243msmean observation cost is documented separately from the historical long-run sampler. Do not invent continuous native history for the five old runs. C4one sampler OSError retained; coverage gaps<=1.157s. Final193tests passed20.09s; Ruff/deps passed,77files formatted; replays1200/300/1615.423protected entries:422unchanged and1intentional benchmark-harness change, no unexpected changes. Timed source archives/frozen policy verified. All eight required artifacts/fiveplots in repeat-20261109T194018Z-c8b6ba59, final verification and readiness/limitations documented. Local Docker experimentation is a next isolated test, not implementation or production readiness.
## Phase 7A implementation plan (2026-11-10)

Audit before edits: read README, architecture, API service/reliability, soak/native profiling guides, settings, lazy lifecycle, workload and dependency configuration. Baseline: 193 tests passed (14.66s); Phase 2/3/4 replays verified 1200/300/1615 rows. Docker client 28.0.4 is present; Linux daemon initially unavailable. Phase 6.2 budget failures and inconclusive leak remain open.

1. Package the unchanged service in a pinned CPU-only Python image; lock dependencies and allowlist the build context.
2. Mount corpus, prepared snapshots, index and frozen policy read-only; provide one-service Compose with local port binding and explicit resource limits.
3. Add configuration/packaging tests and a separate real-container verifier reusing the eight-query, five-method workload. Record build, startup, lifecycle, equivalence and resource observations, including blocked/failed stages.
4. Run integration experiments where Docker/hardware permit, preserve all old outputs, run regression/replay/lint/dependency checks and document exact PowerShell commands and limits.

No production BLAS default, retrieval implementation, frozen routing policy, cloud service or MLflow change is authorized in this milestone. Optional BLAS controls must be explicit experimental overrides.

Phase7A progress: image/Compose/mounts/health/preflight implemented; seven packaging tests and two result-verifier tests pass individually. Existing193tests remain passing (combined200 before the last two additions). First build failed `pip check` on the Linux-only hf-xet dependency; the pin was added and a measured retry built successfully. Docker engine needed starting and sandbox permission for its named pipe. Context transfer347.36kB excludes data/weights. CPU Linux torch2.9.0+cpu requires a separately built cache; Windows index preserved, with no identity bypass. Integration campaign `phase7a-20261110T063328Z-3ae104cc` is in progress: local direct/HTTP and first40Linux query/method pairs match exact IDs/ranks and scores within1e-5. Resource-limit/restart/BLAS checks remain pending until complete reports exist.

Phase7A completed:

- [x] Pinned single-stage CPU image, non-root/read-only serving, source-only build context, graceful command and existing health endpoint.
- [x] Persistent dataset/model/index/policy binds; independently prepared Linux index; environment-based method/logging settings preserve native defaults and explicit/null request behavior. BLAS1 remains a separate opt-in experiment.
- [x] Local plus4GiB/4CPU,2GiB/2CPU and4GiB/4CPU-BLAS1 initial/restart campaigns.840correctness checks passed;560measured attempts,525success/35HTTP429,zero request timeouts/transport errors. All resource counters initialized once per process, all55resource hashes unchanged. All six container stops exited0/no OOM/no automatic restart. Separate active-inference SIGTERM/recreation probe passed; no containers left running.
- [x] All seven required artifacts and three figures, raw failed-attempt/build/lifecycle evidence, actual image/source audit and final portable report. Image412571272bytes; measured cached-base build134.518s. Complete measurements and limits in experiment_results.md and docker_deployment.md.
- [x] Final204tests passed16.52s, lint/format/dependencies and replays1200/300/1615passed. Newly written verification-helper lint/format issue fixed; initial failed check logs retained alongside successful final checks. Earlier423-entry audit417unchanged; five intentional API/config/logging sources plus the prior Phase6.2harness change explain all differences. No earlier protected experiment/retrieval/metric/model/index/policy changed.

Decision: packaging supports the next monitored local MLflow integration experiment. Default C2rejection and2CPUlatency budgets fail in short passes; long-duration Linux growth is unmeasured, leak Inconclusive, deployment readiness unestablished. Linux BLAS1latency improved in one sequential pass, with no demonstrated peak-RSS benefit; do not adopt it as a production default or infer the Windows private-commit effect transfers. No MLflow/cloud/Kubernetes or retrieval/policy rewrite was implemented.

## Phase 7B implementation plan (2026-11-10)

Audit before implementation: inspected README/results/container guide, existing persistence/replay interfaces, Phase2/3/4/6.1/6.2/7Areport schemas and Compose storage. Baseline204tests passed; replays1200/300/1615verified. Preserve algorithms, metrics, frozen policy, BLAS defaults, old outputs and API independence.

1. Add an optional lazy MLflow client adapter after local result persistence, with explicit CLI flags and failure receipts outside canonical artifacts.
2. Normalize saved evaluation and operational reports separately; log selected parameters/real metrics and filtered artifacts, preserving source hashes/timestamps. Historical imports never infer and completed imports deduplicate by source bundle identity.
3. Add an opt-in local MLflow Compose profile with SQLite/backend and artifact volumes, loopback publication, health/restart configuration and no API dependency.
4. Add mock-based tests and a separate real-server integration verifier for new smoke tracking, historical equality/dedup, artifacts, restart/recreation and API behavior with MLflow stopped. Run regressions/replays/lint/dependencies and document actual outcomes/commands.

Phase 7B progress: optional lazy client and centralized saved-result adapters implemented; tracking flags added only at CLI persistence boundaries. Compose has a digest-pinned MLflow 3.5.1 profile, loopback port 5000, independent SQLite/artifact volumes and no API dependency. Fourteen authored/mock tests cover disabled/missing/unavailable clients, exact values, filtering/redaction, source mutation, failed retries/dedup and host/container metadata separation.

Docker's engine returned HTTP 500 after image pull; restarting Docker Desktop restored operation. Initial real uploads exposed the pinned SDK's fluent-URI artifact behavior and Windows stdout emoji encoding; scoped URI restoration and supported print suppression fixed both. A URI/environment-assignment redaction precedence test also failed and was fixed. All attempts/receipts remain preserved.

The v1 pilot completed but host/container Python attribution was refined before finalization. Current v2 identity includes normalized parameters/metrics as well as source/artifact hashes, so changed adapters do not silently reuse stale metadata. Final v2 integration `phase7b-20261110T073341Z-12c807e3` completed: six historical sources, 124 method/condition children, exact parameter equality and 3732 numeric metric comparisons (including repeated checks after recreation), duplicate parent reuse, unchanged persisted artifact hash and 66 unchanged canonical source files. A new tracked smoke succeeded. All five API methods matched preserved rankings/scores/policy with MLflow stopped; an enabled-but-unavailable tracking call still saved and completed its local smoke benchmark. MLflow was recovered for UI inspection, API stopped. Final suite/replays and an additional tracked public BM25 subset are recorded below when finalized.

Phase 7B completed:

- [x] Optional client, safe post-persistence CLI hooks and separate evaluation/operational records with provenance, timestamps, selected artifacts and missing-value handling.
- [x] Local Compose tracking profile, health/UI HTML+JS, loopback/Host checks, independent persistent backend/artifact volumes and no API dependency.
- [x] Six historical parents /124children; two new parents /two children (authored smoke and public25-query BM25 subset). Repeated imports reused parent IDs.3738numeric equality comparisons passed, plus exact parameter equality. No historical inference rerun.
- [x] Restart/recreation retained the tested artifact hash and all records. Outage did not fail offline evaluation or five-method API compatibility.66canonical source files and55prepared resource files unchanged; only four CLI files differ from the64-file Phase7Asource audit, all expected hooks.
- [x] Final218tests passed16.69s; lint/format/dependencies passed; replays1200/300/1615passed. Failures/trials, actual environment freeze, source/report manifests, client/server versions and known limits documented. MLflow remains healthy at localhost5000; API stopped after probe.

Decision: this completes local experiment tracking integration, not scientific retesting, stable service readiness, public hosting or cloud deployment. Preserve the earlier failed resource targets and inconclusive memory status. No algorithm, frozen policy or BLAS default changed; no dataset/model/cache upload or autologging was introduced.
