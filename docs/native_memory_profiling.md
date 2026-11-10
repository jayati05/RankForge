# Phase 6.2 native memory profiling and repeatability

This diagnostic milestone extends the Phase6.1 harness with an explicit `--native-profile` option. The API, retrievers, frozen policy, queue/admission limits and model adapters remain unchanged. The original harness source is archived in `experiments/results/phase6_2/audit/api_soak_before.py.txt`; all prior artifact/model/index/source hashes are audited separately. The per-run report keeps the compatibility filename `phase6_1_report.json`, but its run ID and directory identify Phase6.2.

Run from the repository root in Windows PowerShell using the prepared offline models, dense cache, corpus and frozen policy. No download is required:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
# Short functional check; not long-duration stability evidence
.\.venv\Scripts\python.exe -m app.evaluation.api_soak --native-profile --duration 10 --clients 1 --stress-duration 0 --warmup 5 --idle 5
# Two independent C1, two C2, one intentional C4 stress; expect roughly100minutes
.\.venv\Scripts\python.exe -m app.evaluation.native_repeat
# After serving measurements stop, set the directory printed by native_repeat
$campaign = "experiments/results/phase6_2/REPLACE_WITH_REPEAT_DIRECTORY"
.\.venv\Scripts\python.exe -m app.evaluation.native_controls --mode all --output "$campaign/inference_controls"
.\.venv\Scripts\python.exe -m app.evaluation.native_analysis "$campaign"
```

The full campaign launches fresh serving processes sequentially in order C1/C2/C1/C2/C4. Each normal workload is900s; stress300s; every run has60s warmup,300s idle,1s sampling,30s request timeout,seed42 and the exact Phase6.1 equal-five-method200-request cycle over the same eight queries. Cold health, all-resource initialization, direct/API reference checks, warmup, initial checks, steady workload, stopped snapshot, actual idle, post-idle checks and graceful shutdown are retained. Duration-based closed-loop submissions mean the number of completed cycles and per-method attempts can differ. Already submitted requests drain after the deadline; measured elapsed/drain overruns are recorded. No independent run shares a serving process or model cache in memory; the immutable disk embedding cache is reused.

The unchanged local research budgets are2048MiB peak RSS,2MiB/min late RSS slope in the last120s after at least240s observed,<=5% normal429,100% intentional-stress429 allowance,0timeouts/cancellations/other errors and<=5000ms successful-requestp95. Private-commit trends are additional observations, not an invented new pass threshold. Failed budget checks take precedence over unknown checks; unknown growth alone is inconclusive. Completion is separate from budget success. Two repeats provide descriptive ranges/sample standard deviation, not confidence intervals or production reliability estimates.

`GetProcessMemoryInfo` captures working set/private commit from the actual Uvicorn serving PID parsed from its startup log and independently confirmed in child snapshots. The Windows Python launcher PID is retained separately and never substituted for the server. `VirtualQueryEx` walks committed/reserved regions and summarizes private/image/mapped kinds at stopped lifecycle checkpoints in this campaign; full mapped-file inventories/largest regions are retained there. [Microsoft documents](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualqueryex) that copy-on-write image/mapped pages retain their mapping type, so region totals cannot be equated with PrivateUsage. Reserved virtual address space is neither committed bytes nor resident bytes. [Toolhelp thread enumeration](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-thread32first) counts all process OS threads. The five repeated workload runs use the original1s RSS/private sampler. A module-wiring issue left continuous native-region/thread instrumentation inactive; their thread-count time series is explicitly sparse lifecycle observations with reply-publication timestamps, not invented1s values. An additional short post-fix smoke validates continuous native sampling separately. OS observations are sequential, not an atomic instant.

Configuration is inspected once per actual encoder/scorer inference thread and at control checkpoints: PyTorch intra-/interop limits, threadpoolctl BLAS/OpenMP runtime/version/count, safe native-thread environment keys, PyTorch build, logical tensor/index bytes, Python threads/objects, async task classes and tracemalloc started after model load. The pre-adapter `grad_enabled` observation precedes the library's internal inference context: installed sentence-transformers5.1.2 uses `@torch.inference_mode()` on both encode and predict. It is not evidence of autograd retention. NumPy dense-query arrays/results are transient in the existing implementation; the normalized corpus vector matrix is intentionally cached. No allocation-stack/heap-owner trace is available for anonymous regions. Tensor/array logical sizes and traced-Python deltas cannot fully attribute native buffers, tokenizer/Rayon state, OpenMP/BLAS pools or allocator fragmentation/retention. Never subtract tracemalloc from RSS/commit to label the remainder a leak.

The separate `native_control` probe uses authored float32 normalized5183x384 matrix/vector data,seed42,100identical matrix-vector calls. Fresh processes differ only in OPENBLAS_NUM_THREADS1vs20; actual runtime configuration, before-import/import/array/kernel milestones, PIDs and scores are retained. It evaluates native import/kernel behavior, not IR effectiveness. Use a child-specific environment, never change the service environment for the repeatability campaign. `native_operator --method hybrid|reranked --output NEW_DIRECTORY` separately initializes all resources identically, warms five calls, repeats the same fixed query40times, verifies exact IDs/ranks and scores within1e-5, then profiles one additional CPU call with PyTorch operator-memory counters. Only the selected pipeline differs; profiler output is allocation traffic/net operator counters, not a retained native heap or RSS accounting. The profiler may add memory itself and cannot attribute blocks allocated before its scope. These probes run outside timed HTTP campaigns. Neither forces GC or allocator release.

Each campaign exports native_memory_summary.json, private_commit_timeseries.csv, rss_timeseries.csv, thread_count_timeseries.csv, repeated_run_comparison.csv, request_outcomes.csv, memory_attribution.json and phase6_2_report.json, plus five plots. All attempts including429/timeouts/cancellations/errors remain in outcomes and throughput denominators. Successful and429latencies are separate; per-method summaries remain in every source report. Exact commands, source hashes, safe configuration, hardware/software versions, PIDs and lifecycle/correctness evidence are preserved. See experiment results for measured outcomes, failures, observer effects and the deployment-readiness assessment.


A relevant upstream mechanism is documented in [OpenBLAS0.3.29 memory.c](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.29/driver/others/memory.c): its Windows buffer backend commits and reserves virtual memory; `blas_memory_free` marks pooled buffers unused, while shutdown releases allocations. This supports a possible workspace-pool explanation for the controlled1vs20-thread private-commit difference. It does not prove the exact allocation owner/build path for every anonymous region in the installed wheel or explain later serving-RSS trends. No shutdown/trim call was injected into the running service.


The inference-control runner also repeats the identical reranked condition with child-only OPENBLAS_NUM_THREADS=1. Compare that to the default-thread reranked condition; actual BLAS counts are recorded, rather than assumed. This additional one-variable control tests whether the import/kernel finding transfers to the prepared retrieval pipeline. It is not an adopted serving configuration or a new HTTP benchmark. For a standalone C4 retry in PowerShell use `--clients= --stress-duration 300` to preserve an empty normal-stage list; the full `native_repeat` command already passes that argument safely through Python subprocesses.


## Completed run and next commands

The completed campaign is `experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59`. Its clean primary inference controls are selected by analysis_inputs.json; the interrupted original attempt remains in inference_controls with a validity note. The corrected post-campaign smoke has97/97native samples and99passing correctness checks. Its map-census observation averaged198.243ms, maximum484.663ms: sampling is configured1s plus observer duration, not exact1Hz. Current full reruns therefore use the corrected, more expensive instrumentation; match source/observer configuration before comparing timing. The old measured source versions are archived under audit/timed_sources; they are not silently replaced in the historical report.

```powershell
# Inspect measured results; this does not rerun the long campaign
Get-Content experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/repeated_run_comparison.csv
Get-Content experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/final_validation.json
# Repeat functional/regression checks
.\.venv\Scripts\python.exe -m pytest -q
# Rebuild derived plots/summary from preserved observations (validation logs remain separate)
.\.venv\Scripts\python.exe -m app.evaluation.native_analysis experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59
```

The existing final report also embeds verification evidence and has a preserved portable copy under experiments/reports. Regenerating analytical outputs recalculates observations; it does not re-run or certify the old verification commands. No serving limit/default/thread setting was tuned, and no Docker implementation was added.
