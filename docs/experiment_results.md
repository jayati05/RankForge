# Verified Phase 1 results

Run `20261025T172326Z-fa70553c4144`: synthetic smoke dataset only, 6 documents, 4 queries, 4 judgments, k=10.

| Metric | Observed value |
| --- | --- |
| Recall@10 | 1.0 |
| Precision@10 | 0.1 |
| MRR@10 | 1.0 |
| nDCG@10 | 1.0 |
| Mean first-pass search latency (ms) | 0.03915000706911087 |
| Index build time (seconds) | 9.430013597011566e-05 |

This verifies software plumbing; it provides no evidence of general retrieval quality or production performance. Precision uses denominator 10 even though the corpus contains only 6 documents. No public benchmark, statistical significance, latency percentiles or memory measurement is claimed.

Raw outputs: `experiments/results/20261025T172326Z-fa70553c4144/` contains benchmark_summary.csv, per_query_results.csv, rankings.json and run_metadata.json. Artifacts are ignored by Git; these measured values are preserved here.

Checks: pytest 29 passed after fixing a PowerShell Unicode fixture-encoding error (initial run: 28 passed, 1 failed). Ruff lint passed. Formatting check initially identified the corrected test file; formatting was applied and rechecked. See decisions.md for final checks and commands.

## Public BEIR SciFact BM25 milestone

The original synthetic run above is preserved unchanged and is not combined with the following public-data evaluations. Parameters were frozen before execution; no tuning was performed.

| Evaluation | Queries | Judgments | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Mean search ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SHA256 subset, seed 42 | 25 | 26 | 0.640000 | 0.068000 | 0.551333 | 0.572347 | 14.289820 |
| Full BEIR test split | 300 | 339 | 0.759833 | 0.083333 | 0.598321 | 0.633399 | 14.278240 |

Both runs search all 5,183 documents. Raw artifact directories:

- `experiments/results/scifact/20261025T174241Z-b4b56da62891/` (index build: 0.522485 seconds).
- `experiments/results/scifact/20261025T174246Z-5ea5ddf8cefd/` (index build: 0.520429 seconds).

Each directory contains `benchmark_summary.csv`, `per_query_results.csv`, `rankings.json`, and `run_metadata.json`. The [machine-readable report](../experiments/reports/scifact_bm25.json) preserves both exact summaries and complete run metadata outside the ignored results directory. The report contains IDs/hashes/settings, not document text. Raw artifacts and the public dataset remain local and ignored.

Execution environment: Python 3.11.5, rank-bm25 0.2.2, NumPy 2.2.6; Windows build 26200, Intel64 Family 6 Model 154 Stepping 3, 20 logical CPUs reported. Indexing and search ran locally on CPU. All BM25 settings match the initial smoke implementation: k1=1.5, b=0.75, epsilon=0.25; title+body; NFKC/casefold identifier tokenizer; document-ID tie breaking. Metadata records software, machine, input and code hashes.

Archive identity: published MD5 `5f7d1de60b170fc8027bb7898e2efca1`; downloaded SHA256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`. Source and license attribution are in [data provenance](../data/README.md).

Validation: all 300 per-query metrics were recomputed from saved rankings and original qrels; all matched, including macro summary. The 25-query subset matched the same rankings in the full run; train/test query IDs were confirmed disjoint. Test and lint outcomes are recorded in decisions.md.

Limitations: unjudged documents have zero evaluation gain, without asserting known irrelevance; sparse relevance judgments can undercount valid retrievals. The 25-query sample is a diagnostic subset of test, not independent validation. Full test results have now been observed and must not be used for tuning. No inferential statistics, method comparisons, memory measurements or steady-state latency conclusions are supported. These results are specific to this tokenizer/BM25 implementation and are not a claim to reproduce published BEIR engine results.

Download failure recorded: direct Python HTTPS returned `SSL: CERTIFICATE_VERIFY_FAILED`. Windows PowerShell download succeeded with certificate verification, and local checksum-verified import completed. No benchmark execution failed.

## Phase 2 initial 25-query diagnostic comparison

Run `phase2-20261025T175857Z-08cbfb3c0287`, fixed seed 42, same 25 test queries as the Phase 1 subset, all 5,183 documents and 26 judgments. This diagnostic sample is not a separate validation set. Models and settings were frozen before execution.

| Method | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Mean ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.640000 | 0.068000 | 0.551333 | 0.572347 | 17.201 | 15.986 | 25.648 |
| dense | 0.680000 | 0.072000 | 0.551333 | 0.585675 | 10.125 | 9.810 | 11.781 |
| hybrid | 0.720000 | 0.076000 | 0.524159 | 0.572376 | 26.758 | 26.368 | 37.873 |
| reranked | 0.680000 | 0.072000 | 0.539048 | 0.573618 | 879.573 | 898.132 | 923.640 |

Raw artifacts: `experiments/results/phase2/phase2-20261025T175857Z-08cbfb3c0287/`. [Exact machine-readable report](../experiments/reports/phase2_scifact_25.json). Replay verified all 100 per-query rows. BM25 rankings and scores match the preserved Phase 1 subset exactly. Reranking used 20 candidates/query; on this small sample it increased nDCG slightly versus hybrid but reduced Recall@10 from 0.72 to 0.68, at substantially greater latency. It improved per-query nDCG for 4 queries, degraded it for 4 and preserved it for 17. This is a mixed, descriptive result.

Query `1303` is an observed reranking improvement (nDCG difference +0.569323 versus hybrid); query `54` is an observed regression (-0.666667). Full candidate positions and metrics are in the report and failure_cases.json. These examples were selected from measured extremes, not authored illustrations.

These are warmed, independent complete-pipeline timings: source depth 50, RRF constant 60, candidate pool 20, CPU with 4 torch threads, batch 16, no cross-pipeline query/result reuse. The 25-query p95 is especially unstable. Model loading was excluded (dense 5.549185s including first library imports, cross-encoder 0.138147s); cached dense setup 0.107680s, BM25 build 0.500531s. The actual original document-embedding build was 95.411672s and is retained in cache metadata. Cache setup is not a fresh embedding build measurement.

The full 300-query comparison completed; its measured results follow below. See [Phase 2 methodology](phase2.md) for model revisions, truncation, timing and exposed-test limitations.

## Phase 2 full SciFact test comparison

Run `phase2-20261025T180228Z-bdac57e0c104`: all four methods evaluated the same 300 test queries, 339 judgments and complete 5,183-document corpus. Configuration was unchanged after the diagnostic run. No parameters were tuned on test. [Exact machine-readable report](../experiments/reports/phase2_scifact_full.json).

| Method | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Mean ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.759833 | 0.083333 | 0.598321 | 0.633399 | 19.145 | 18.285 | 30.848 |
| dense | 0.783333 | 0.088333 | 0.604725 | 0.645082 | 12.125 | 11.669 | 16.447 |
| hybrid | 0.822111 | 0.091000 | 0.637352 | 0.676569 | 30.161 | 29.284 | 43.285 |
| reranked | 0.834722 | 0.093000 | 0.668989 | 0.700349 | 938.427 | 944.082 | 1041.990 |

Reranking increased average nDCG@10 by **0.023780** and Recall@10 by **0.012611** versus hybrid. Its mean complete-pipeline latency was **31.11 times** hybrid's (denominator: hybrid mean latency), an additional **908.267 ms/query**. These are absolute metric differences, not percentage-point claims about significance. Dense had a smaller observed quality gain over BM25 with lower measured query latency in this run, but required model loading and a substantially more expensive index build. Hybrid improved observed mean quality over either standalone method at additional query cost. The best average scores do not imply wins on every query or establish statistical superiority.

Reranking improved per-query nDCG for **65** queries, degraded it for **53**, and left it unchanged for **182**. All 300 queries had some disagreement in the full ranked-ID lists. There were **45 judged-positive query/document pairs** absent from the 20-document hybrid pool; **33 queries** had no judged-positive candidate in that pool. Reranking cannot retrieve those absent candidates. This is an observable candidate limit, not an inferred root cause.

Model loading (excluded from timed queries): sentence encoder **5.500070s**, cross-encoder **0.139383s**. The first load includes imports. Index setup: BM25 build **0.589433s**, dense cache hit/load/validation **0.100985s**; shared index setup for hybrid/reranked **0.690418s**. The original document embedding/normalization build was **95.411672s**, recorded in the cache; no cached setup time is presented as embedding construction. The compressed index file is 7,409,433 bytes, which is an artifact size, not a peak-memory measurement.

Settings: CPU, 4 torch threads, batch 16; source depth 50, RRF constant 60, rerank pool exactly 20 for every query; dense maximum 256 tokens and cross-encoder pair maximum 512. One warmup per complete pipeline; independent searches in rotated order, with no shared per-query results. Full-run reranker incremental scoring/sorting averaged **908.279ms**; end-to-end latency includes retrieval. Exact versions/hardware/model revisions and all initialization/warmup times are in the report. p50/p95 are descriptive single-pass statistics, not throughput or statistical-significance results.

Raw full-run directory: `experiments/results/phase2/phase2-20261025T180228Z-bdac57e0c104/`. Files: `benchmark_summary.csv`, `benchmark_summary.json`, `per_query_results.csv`, `rankings.json`, `candidate_pools.json`, `failure_cases.json`, `run_metadata.json`. Input hashes, query IDs, candidate provenance, failure observations and both CSV/JSON summaries replay successfully. All **1,200** per-query rows were verified. The new BM25 ranked IDs and scores exactly match the original preserved full Phase 1 run.

### Strongest observed examples

The following cases were selected automatically from actual output by maximum gain/loss in per-query nDCG@10, breaking ties by query ID. Null top-10 positions mean outside the cutoff; they do not assert a missing corpus document. These selected extremes are not representative samples.

| Comparison | Query ID | Relevant document | Observed position change | nDCG difference |
| --- | --- | --- | --- | --- |
| Hybrid to reranked improvement | 1241 | 4427392 | Pool rank 20 to 1 | +1.000000 |
| Hybrid to reranked regression | 94 | 1215116 | Pool rank 1 to 9 | -0.698970 |
| BM25 to dense improvement | 1088 | 37549932 | Outside top 10 to rank 1 | +1.000000 |
| BM25 to dense regression | 1024 | 5373138 | Rank 1 to outside top 10 | -1.000000 |

Query 1241 shows a document already present at the candidate cutoff being promoted to rank 1. Query 94 shows reranking harming an already-high-ranked judged-positive document. Queries 1088 and 1024 demonstrate opposite outcomes for dense versus lexical retrieval. No cause such as paraphrase understanding or domain mismatch is asserted from these observations alone.

### Final verification and limitations

Final full pytest: **93 passed in 1.44s**. Ruff lint: **All checks passed**. Ruff format check: **26 files already formatted**. pip check: **No broken requirements found**. Focused dense tests: 17 passed; RRF/reranking tests: 17 passed. Ordinary tests use authored vectors/model doubles and make no model or large-data downloads.

The documented real-model synthetic smoke command also completed separately at `experiments/results/phase2-smoke/phase2-20261025T180750Z-edfa75e06081/`; replay verified all 16 per-query rows. It uses k=3, pool=3, depth=5 and six synthetic documents, so its results are software verification only and are not compared with the public benchmark.

The only failed Phase 2 command was an initial model-download invocation made before installation had finished (missing huggingface_hub); retry succeeded. No comparison or indexing run failed. The prior Phase 1 direct-Python dataset HTTPS certificate failure remains documented; model downloads used system trust with certificate verification enabled. Full public comparison and smoke artifacts have status=complete; old synthetic/public artifacts were not overwritten.

Limitations remain: incomplete qrels and zero unjudged evaluation gain; one exposed biomedical test set; short general-purpose/MS MARCO model transfer and text truncation; single CPU/machine and one warmed observation per query; no confidence intervals, peak-memory study, tuning, ablations or concurrency measurements. Phase 2 implements descriptive disagreement evidence, not the complete Phase 3 taxonomy/category/routing research program.

## Phase 3 Failure Lab and exploratory routing (2026-10-31)

Run `phase3-20261031T065803Z-72804b2af31a` completed. Historical analysis reuses the replay-verified complete Phase 2 source (300 queries, 5,183 documents, 339 judgments), preserving all source outputs and metric definitions. Live routing comparison uses the exact same 60 internal held-out queries for every baseline and router. The split is 180 development / 60 validation / 60 held-out, SHA256 seed 42, with IDs saved before tuning. It is an internal exploratory partition of an exposed benchmark, not untouched external evaluation.

The finite predeclared rule family was shortlisted on development and selected on validation by nDCG minus 0.1 times historical mean latency in seconds, with a <=50% reranking rate constraint on those selection groups. The selected policy disables both shortcuts (rare threshold 1.01, rerank length zero): hybrid-only. The validation winner had nDCG 0.685828, estimated latency 28.955685ms and utility 0.682933. Full candidate scores and exact tuning label IDs are in tuning_audit.json. Policy hash and ordered events verify freezing before held-out evaluation. No threshold was changed after observing held-out outcomes.

### Fresh internal-held-out measurements

| Method | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Mean ms | p50 ms | p95 ms | Reranker rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.755556 | 0.078333 | 0.573353 | 0.611713 | 19.093 | 18.178 | 33.093 | 0% |
| dense | 0.758333 | 0.081667 | 0.567222 | 0.613868 | 13.807 | 11.479 | 19.563 | 0% |
| hybrid | 0.802778 | 0.085000 | 0.593419 | 0.639072 | 31.107 | 29.910 | 44.030 | 0% |
| reranked | 0.858333 | 0.091667 | 0.647057 | 0.689126 | 896.722 | 906.554 | 983.601 | 100% |
| router | 0.802778 | 0.085000 | 0.593419 | 0.639072 | 31.550 | 29.604 | 51.859 | 0% |

Routing shares: BM25 **0%**, hybrid **100%**, reranked **0%**. The router's mean latency savings versus always-reranked is **96.48%**, using always-reranked mean latency as denominator. This is the hybrid-versus-reranked cost/quality trade-off; it is **not an adaptive router improvement**. Router and hybrid have identical ranks and metrics. Their small measured timing differences reflect independent execution/overhead/noise and do not establish an algorithmic speedup.

| Paired router minus baseline | Mean nDCG difference | Exploratory 95% query-bootstrap interval |
| --- | --- | --- |
| bm25 | +0.027359 | [-0.029661, +0.083862] |
| dense | +0.025204 | [-0.039054, +0.083916] |
| hybrid | +0.000000 | [+0.000000, +0.000000] |
| reranked | -0.050054 | [-0.115427, +0.012506] |

Intervals use 2,000 paired query bootstrap resamples with seed 42, uncorrected for multiple comparisons. They do not capture repeated-run runtime noise or remove prior benchmark exposure. Intervals against BM25, dense and reranked all include zero; no statistically established quality superiority is claimed. The hybrid interval is exactly zero because this frozen policy returns hybrid for every query.

Timing: local CPU, inherited Phase 2 batch 16 / 4 torch threads, dense snapshot and cache reused with verification, source depth 50 / RRF 60 / rerank pool 20, k=10. Models and indexes load once outside query timing. Every complete system receives an authored-text warmup. Five-system order rotates per query; the router executes its selected pipeline independently and includes lexical feature extraction/decision overhead. No historical latency is substituted for fresh live timing. Historical failure CSV timings remain labeled as Phase 2 observations. Full environment/provenance is in run metadata and environment-freeze-phase3.txt.

### Historical failure indicators over all 300 source queries

| Nonexclusive indicator | Query count |
| --- | --- |
| lexical_mismatch | 2 |
| semantic_mismatch | 27 |
| candidate_generation_failure | 40 |
| ranking_failure | 20 |
| reranker_regression | 53 |
| ambiguous_or_underspecified | 0 |
| insufficient_judgments | 277 |
| other_or_unclassified | 5 |

Candidate-generation failure means at least one judged-positive document is absent from the saved 20-document hybrid pool (40 queries); 45 positive query/document pairs are missing, and 33 queries have no positive in that pool. Ranking failure means a judged-positive candidate exists but is below top 10 (20 queries). Reranker regression remains 53 queries. Counts are nonexclusive. The two lexical-mismatch and 27 semantic-mismatch classifications are evidence-backed hypotheses, not known causal explanations. The 277 insufficient-judgment indicators simply flag <=1 positive judgment; they do not mean those 277 query labels are invalid. No public source query has zero judgments, but the analysis retains/flags zero-judgment fixtures explicitly.

Examples are selected from actual output (no hardcoded classifier labels):
- `lexical_mismatch`: query `1`; evidence `{"caveat": "Zero tokenizer overlap and absent from BM25 top-k; not proof of causal failure", "no_shared_tokens_judged_positive_ids": ["31715818"]}`.
- `semantic_mismatch`: query `1024`; evidence `{"bm25_hits_missing_from_dense_topk": ["5373138"], "caveat": "Differential retrieval is observed; semantic cause is not established"}`.
- `candidate_generation_failure`: query `1049`; evidence `{"missing_judged_positive_ids": ["12486491"], "pool_size": 20}`.
- `ranking_failure`: query `1029`; evidence `{"cutoff": 10, "judged_positive_in_pool_below_cutoff": ["13940200"]}`.
- `reranker_regression`: query `1`; evidence `{"delta": -0.3010299956639812, "hybrid_ndcg": 0.3010299956639812, "reranked_ndcg": 0.0}`.

The earlier measured query 1241 promotion (candidate rank 20 to 1) and query 94 regression (rank 1 to 9) remain useful specific observations in the Phase 2 report. No paraphrase, domain-mismatch or ambiguity cause is asserted without additional evidence.

### Artifacts, figures and verification

All required artifacts exist under `experiments/results/phase3/phase3-20261031T065803Z-72804b2af31a/`: failure_cases.csv, per_query_comparison.csv, query_features.csv, failure_summary.csv, system_comparison.csv, routing_decisions.csv, failure_analysis.json, routing_evaluation.json, split_metadata.json. Extra files retain frozen policy, tuning audit, live per-query rows, live rankings, status, source/code hashes and output manifests. The [measured portable report](../experiments/reports/phase3_scifact.json) preserves the summary/splits/policy/provenance outside ignored raw outputs.

Five figures in `plots/` derive only from saved data: ndcg_vs_p50.png, query_delta_distribution.png, failure_categories.png, router_distribution.png, paired_quality_latency.png. Visual review found overlapping hybrid/router labels; their offsets were corrected and figures rendered again from unchanged saved data, with the render-code hash recorded. No inference was repeated and no policy/metric changed for that rendering fix.

Final checks: **104 tests passed in 1.67s**; Ruff lint passed; Ruff formatting **35 files already formatted**; dependency check passed. Phase 2 replay verified **1,200** rows; Phase 3 replay verified **300** live baseline/router rows. Offline tests cover features, missing/empty input, taxonomy/evidence, zero judgments, all router branches/explanations and invocation counts, deterministic splits, forbidden held-out tuning IDs, bootstrap conventions, persistence/replay and preservation of source files. Ordinary tests do not download models/data. All nine required artifacts and five figures were verified; no query/system inference failure occurred.

Resolved implementation/setup issues: sandboxed plot-dependency installation could not reach build dependencies, then approved network installation succeeded; one unused import was removed after lint identified it; plot-label overlap was corrected. No failed experiment is hidden. This run's status is complete.

Limitations: one previously exposed dataset and only 60 internal held-out queries; finite hand-designed rule family and predeclared utility, not optimal routing; sparse qrels and documented zero-gain convention for unjudged documents; source top-k diagnostic overlap does not reconstruct the unrecorded full source pools; short-query and semantic/lexical taxonomy labels are heuristic; single-pass local CPU timings; no external validation, repeated timing study or causal failure claims. The recommended next analytical step is pre-registering a policy on genuine development data and evaluating a different public dataset. Product/API work remains a separate scope.

## Phase 4: external NFCorpus validation (2026-10-31)

Completed run `phase4-20261031T071828Z-46027d7d4330`. [Portable exact report](../experiments/reports/phase4_nfcorpus.json); [methodology and commands](external_dataset_validation.md). All prior experiments remain unchanged.

The exact Phase 3 policy imports without behavior changes: rare threshold 1.01, rerank maximum tokens 0, coverage threshold 0.8. Canonical record SHA256 `31b0cef1d5969d2b79953be683050877038c34fabc4c560b4a1b8be8f90b67f8`, raw policy file SHA256 `68a6b6e0251f0f1b4da471e96e30711053c332d32704a0572483cf5f49213617`. The registration was created at 07:11:51 UTC before external preparation/label parsing at 07:12:48 UTC. No external policy selection, tuning or fine-tuning occurred. Policy/implementation/model identities were verified before and after evaluation.

### Provenance and observed data

Official BEIR NFCorpus ZIP: published MD5 `a89dba18a62ef92f7d323ec890a0d38d`; downloaded SHA256 `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`. The archive is preserved under data/public/archives/nfcorpus.zip. Required files were copied byte-for-byte and their hashes/joins validated; no documents, text, IDs or grades were filtered or rewritten. Version is identified by archive digest. The BEIR card lists CC-BY-SA-4.0, with attribution/license/change notices retained in provenance; original project page timed out, while primary publication and BEIR sources were available.

Observed corpus: **3,633 documents**. Raw queries: **3,237**; train **2,590 / 110,575 judgments**, dev **324 / 11,385**, test **323 / 12,334**. Split query IDs are disjoint and no raw query ID is unaccounted for. The official test split was selected before label access in the protocol; train/dev were checked only for provenance/schema/counts. Test grades contain **11,758 ones and 576 twos**. The existing binary-relevance and exponential graded-nDCG definitions were reused, with zero gain for unjudged documents as a convention rather than a known-irrelevance claim.

### Complete external test results

| Method | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Mean ms | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.140725 | 0.211455 | 0.499257 | 0.294681 | 6.326 | 5.927 | 12.922 |
| dense | 0.154988 | 0.243344 | 0.507654 | 0.317272 | 23.181 | 16.555 | 55.027 |
| hybrid | 0.154521 | 0.238080 | 0.538718 | 0.324066 | 27.903 | 22.956 | 58.607 |
| reranked | 0.161816 | 0.252322 | 0.575036 | 0.351809 | 1896.101 | 1378.073 | 4301.621 |
| router | 0.154521 | 0.238080 | 0.538718 | 0.324066 | 26.119 | 21.772 | 57.153 |

Every system evaluated the exact same 323 query IDs, 3,633 documents and qrels at k=10. Router distribution is BM25 0%, hybrid 100%, reranked 0%; reranker invocation rates are 100% for always-reranked and 0% for all other systems. The frozen router has identical ranked IDs/scores and metrics to standalone hybrid. Its lower independently measured mean latency does not imply an algorithmic speedup: it executes the same hybrid work plus routing overhead in a different rotated timing slot.

| Within-NFCorpus paired difference | Mean nDCG difference | 95% paired-query bootstrap interval |
| --- | --- | --- |
| hybrid_minus_bm25 | +0.029385 | [+0.016479, +0.042194] |
| reranked_minus_hybrid | +0.027743 | [+0.015335, +0.040171] |
| dense_minus_bm25 | +0.022590 | [-0.000953, +0.045779] |

Intervals use 2,000 query resamples, seed 42, and are uncorrected for multiple comparisons. Hybrid-minus-BM25 and reranked-minus-hybrid intervals exclude zero in this dataset/pass; dense-minus-BM25 spans zero. These are within-collection quality observations, not universal-superiority claims or repeated-timing confidence intervals.

Reranking improved **129**, degraded **75**, and preserved **119** query-level nDCG values. Its mean gain versus hybrid was **0.027743**, costing **1868.198ms/query** and **67.95 times** hybrid mean latency. Router mean latency savings versus always-reranking was **98.62%**, with the same quality loss as hybrid. Whether that fixed compromise is appropriate depends on application quality/latency requirements; it is not evidence of an adaptive policy win. No external threshold was changed to favor any result.

### Comparison with SciFact

| Method | SciFact full nDCG@10 (300 queries) | NFCorpus nDCG@10 (323 queries) | NFCorpus minus SciFact |
| --- | --- | --- | --- |
| bm25 | 0.633399 | 0.294681 | -0.338717 |
| dense | 0.645082 | 0.317272 | -0.327810 |
| hybrid | 0.676569 | 0.324066 | -0.352502 |
| reranked | 0.700349 | 0.351809 | -0.348540 |

These are absolute descriptive subtractions across different query sets; they are not paired differences, causal drops or percentage improvements. Full Recall/Precision/MRR/nDCG deltas are retained in generalization_analysis.json. NFCorpus uses graded judgments and averages about 38.19 judged positives per query versus SciFact's 1.13. A cutoff of ten makes recall much harder to saturate on densely judged queries. Grade and task differences also affect nDCG and category rates. The frozen router's SciFact reference is the separate Phase 3 60-query internal slice (nDCG 0.639072), not an invented full-300-query router benchmark; NFCorpus router nDCG is 0.324066.

Observed mean nDCG ordering in both collections was reranked > hybrid > dense > BM25. Hybrid and reranking therefore retain the direction of mean nDCG improvements on this second task. This ordering does not imply dominance over every metric/query: on NFCorpus, dense Recall@10 0.154988 and Precision@10 0.243344 exceed hybrid's 0.154521 and 0.238080. Reranking had the largest observed average quality but substantially higher latency. The frozen policy remains a fast hybrid operating point with lower quality than reranking, not a learned adaptive generalization result.

Both datasets are broadly biomedical, though SciFact claims and NFCorpus lay medical/nutrition topics differ. One additional task cannot establish broad or production-scale generalization. Differences in query phrasing, judgment density, truncation and runtime conditions are possible explanations, not isolated causal effects. Pretrained-model overlap with these collections was not audited. Historical SciFact timing and current NFCorpus timing occurred in different runs/dates; even on the same CPU with the same settings, they are not controlled cross-dataset performance experiments.

### Failure patterns and examples

| Nonexclusive indicator | SciFact count / 300 | NFCorpus count / 323 |
| --- | --- | --- |
| lexical_mismatch | 2 | 232 |
| semantic_mismatch | 27 | 133 |
| candidate_generation_failure | 40 | 307 |
| ranking_failure | 20 | 150 |
| reranker_regression | 53 | 75 |
| ambiguous_or_underspecified | 0 | 193 |
| insufficient_judgments | 277 | 23 |
| other_or_unclassified | 5 | 2 |

NFCorpus candidate-pool misses occur on 307 queries when any judged-positive ID is missing; this does **not** mean all 307 queries have no useful candidate. Exactly 74 queries have no positive in the pool, 11,153 positive query/document pairs are outside the pool, and 144 queries have more than 20 judged-positive documents, making full recovery by a fixed 20-document pool impossible. These facts distinguish candidate limitations from the much larger judgment denominator. The shared ranking-failure rule triggers on 150 queries with a positive in the pool below hybrid top ten. Short-query and lexical/semantic indicators are heuristics, explicitly marked inferred; the higher counts do not establish causal semantic/lexical failure or universal model degradation.

Automatically selected strongest reranking changes from actual output: `PLAIN-3026` nDCG delta **+0.684535** versus hybrid; `PLAIN-1288` delta **-0.560976**. Complete document IDs, judgments, ranks, per-query counts and supporting evidence are in failure_analysis.json, rankings.json and candidate_pools.json. These are selected extremes, not representative or manually assigned labels.

### Timing, persistence and verification

CPU execution reused the exact pinned model/library identities, batch 16, four torch threads, source depth 50, RRF 60, rerank pool 20, dense limit 256 and cross-encoder pair limit 512. No model was downloaded anew. Separate dense document embedding construction took **206.524248s**; this comparison used a cache hit, whose setup/validation took **0.262224s**. BM25 index build took **1.315552s**. Model loading was excluded from query latency: sentence encoder **10.946692s** (including first imports), cross-encoder **0.473501s**. Corpus-statistics setup was separately recorded. Warmup and independent rotated five-system execution match Phase 3; router timings include feature/decision overhead. Reranking latency and actual input counts are in live rows, and retrieval-component latency is explicitly a total-minus-rerank estimate.

All nine required artifacts and five saved-data figures exist under `experiments/results/phase4/phase4-20261031T071828Z-46027d7d4330/`. Extra files preserve live rows, rankings, candidate pools, detailed failure/routing evaluation, copied exact frozen policy/protocol and hash manifests. The portable report preserves exact measured values/provenance outside ignored raw outputs. Plots show nDCG versus p50 latency, all metric values, per-query rerank changes, failure fractions across collections and four-method SciFact/NFCorpus score comparisons. Figures were visually inspected; no manually entered metrics were used.

Final tests: **112 passed in 7.13s**; Ruff lint passed; formatting **40 files already formatted**; pip check found no broken requirements. Phase 2 replay verified **1,200** rows, Phase 3 **300**, Phase 4 **1,615**. Replay verifies input/output hashes, exact policy/implementation identity, matching five-system query sets, metric/candidate/route compatibility, aggregates and within-dataset bootstrap/generalization reports. Offline tests cover raw/checksum/schema/graded joins, frozen import, a policy mutation during evaluation rejected with explicit failed status, no tuning calls, persistence and backward compatibility. The intentional failed test fixture is not a failed public experiment.

The first sandboxed dataset download was refused (WinError 10061); approved retry succeeded with certificate verification enabled. Two unused imports were removed when lint identified them. No public evaluation failed, no original output was overwritten, and no unresolved critical verification remains. Limitations include single-pass runtime variability (especially p95), different judgment/task distributions, one additional biomedical collection and a nonadaptive frozen fallback. An engineering hardening/API phase can now use these measured operating points, with an explicit quality/latency requirement and no adaptive-router claim; repeated timing and broader-domain validation remain analytical follow-up work.


## Phase 5: local service engineering verification (2026-10-31)

This milestone adds HTTP orchestration only. Existing retrieval/metrics, frozen routing policy and public quality results above remain unchanged. The API loads corpus text only and computes no relevance metrics. Final explicit live verification ran the actual `python -m app.api` process on an ephemeral loopback port with prepared SciFact corpus (5,183 documents), read-only dense cache, pinned local models, CPU/four threads, batch 16, source depth 50, RRF constant 60 and rerank pool 20. Model-library offline flags were enabled for verification; no model/data download or index construction occurred.

Final run: [portable exact report](../experiments/reports/phase5-20261031T083003Z-c4676089f670.json), raw `experiments/results/phase5/phase5-20261031T083003Z-c4676089f670/`. Saved files include real HTTP response payloads, OpenAPI, before/after health, server logs, software/configuration/source hashes, old-artifact SHA256 manifest, environment freeze and final checks. The earlier successful run `phase5-20261031T082604Z-0f3d47060a77` remains separate; the final run followed schema/documentation-example fixes, not score tuning.

All five methods returned exactly the preserved Phase 2 top-10 document IDs/ranks for **one** existing public query, SciFact `1241`; scores agreed within 1e-5. Router matched hybrid and returned exact canonical policy `31b0cef1d5969d2b79953be683050877038c34fabc4c560b4a1b8be8f90b67f8`. This is a compatibility check, not another quality evaluation or evidence of universal equivalence across all inputs. Every required endpoint, docs/ReDoc HTML and OpenAPI returned 200; comparison produced ten method pairs. Invalid request, oversized body, missing route and wrong verb returned safe 422/413/404/405 responses. Offline tests additionally exercise initialization/inference boundaries, missing resources/dependencies, 429 capacity, 500 errors, concurrency and shutdown.

| Method | First HTTP ms | First-use initialization ms | Warm mean HTTP ms | Warm mean complete-pipeline ms |
| --- | --- | --- | --- | --- |
| bm25 | 646.189 | 626.979 | 14.996 | 12.813 |
| dense | 17.805 | 0.000 | 17.307 | 15.028 |
| hybrid | 33.863 | 0.000 | 34.021 | 31.443 |
| reranked | 1527.023 | 148.429 | 1328.148 | 1325.362 |
| router | 6617.159 | 6572.549 | 34.187 | 31.817 |

Each warm mean uses **three sequential requests for the same query**; no percentiles, confidence intervals, throughput or representative workload claims are made. Request order was BM25, router, dense, hybrid, reranked. Router's first call paid for encoder/index/features/policy initialization; later standalone dense/hybrid calls reused those resources, so their first initialization is zero. First BM25 initialization includes corpus/index setup; first reranked initialization includes cross-encoder construction. Each resource initialized once. The router left the cross-encoder unloaded until reranked was explicitly requested. First-use and warm differences include shared initialization effects and runtime noise; the figures cannot be equated with prior independently rotated benchmark timings. The whole reranking pipeline includes its upstream retrieval.

HTTP wall time covers POST through full response receipt, excluding subsequent client JSON decoding. Service pipeline time excludes separately measured initialization, response JSON encoding and network transit; rerank time is a component, not additive. The guide distinguishes payload construction, queue, total service and middleware durations. The API returns full document text, so response sizes/timings depend on requested results. These one-query observations do not establish production readiness or compare relevance quality.

Final complete suite: **159 passed in 7.82s** (112 existing + 47 new), no warnings; Ruff lint passed; **59 files already formatted**; pip check passed; editable API installation succeeded. Phase 2 replay **1,200** rows, Phase 3 **300**, Phase 4 **1,615**. All **134** protected existing experiment/config/model/index files in the final manifest remained unchanged; the manifest includes the earlier Phase 5 portable report as well as Phase 1-4 artifacts and local models/indexes. Current API/service/verifier source hashes match the final HTTP run. Original datasets are additionally guarded by the unchanged replay workflows. Environment is saved separately in `docs/environment-phase5-freeze.txt`; prior environment/results files were preserved.

One intermediate suite failed its recorded OpenAPI example contract (1 failed, 155 passed); nullable routing/default handling was corrected and the complete suite passed. Dependency setup encountered package-index DNS and missing wheel build-tool errors; explicit public PyPI preparation and offline editable installation resolved them. AnyIO was pinned to the tested 4.11.0 compatibility release after an upstream deprecation warning; warnings are not suppressed. No public HTTP verification run failed, and there is no unresolved critical check failure.

Limits: locally validated single CPU process with serialized searches and bounded queue, one sampled public query, no concurrent live load benchmark, no memory profiling/hard inference cancellation/upload deadline, no fresh-environment install or browser/CDN rendering verification. Lifespan cleanup is tested; the real-HTTP helper stops its own Windows subprocess by termination after checks. No auth, dashboard, container or cloud deployment is included. Recommended next milestone: repeatable mixed-query concurrency and timeout/resource-budget tests, with an explicit latency target, before a dashboard or deployment. [Exact Windows commands and service contract](api_service.md).


## Phase 6: mixed-query API reliability (2026-11-09)

Completed run: [portable exact report](../experiments/reports/phase6-20261109T153436Z-0e8e7412820d.json), raw `experiments/results/phase6/phase6-20261109T153436Z-0e8e7412820d/`. The actual Windows API process (PID 8640) was measured separately from the venv launcher (PID 12068). This matters: an earlier timing-complete pass sampled the launcher and its ~4MiB memory result is invalid, explicitly marked failed/reviewed rather than treated as server memory.

Same seeded 100-request schedule per level, eight public/authored queries, five methods with 20 attempts each, seed 42, ten warmup requests, k=10, CPU/four threads, batch 16, depth 50, RRF 60, rerank pool 20. The engine remains serialized with two waiting slots and a two-second queue timeout. The HTTPX workload timeout is 30s; cold initialization has a separate 900s allowance. All levels 1/2/4/8/16 were actually attempted.

| Clients | Success / 100 | Failed (429) | Timeouts | Successful/s | All attempts/s | Success mean ms | p50 ms | p95 ms | p99 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 100 | 0 | 0 | 2.156 | 2.156 | 463.600 | 43.732 | 2263.466 | 2324.969 |
| 2 | 97 | 3 | 0 | 4.437 | 4.575 | 388.176 | 57.471 | 1135.067 | 2203.120 |
| 4 | 54 | 46 | 0 | 4.939 | 9.146 | 557.498 | 85.893 | 1860.309 | 1962.263 |
| 8 | 15 | 85 | 0 | 12.213 | 81.423 | 239.852 | 77.189 | 965.278 | 971.322 |
| 16 | 14 | 86 | 0 | 11.133 | 79.522 | 203.433 | 83.588 | 914.928 | 916.097 |

**280 successful / 500 measured attempts; 220 failed, all HTTP 429; zero measured-workload timeouts.** In the separate cold-initialization test, 1/4 succeeded and 3/4 were rejected with 429 while a single encoder/index initialization proceeded. Warmup/correctness/probes are outside these 500 attempts. Error summaries retain all non-successes; the explicit timeout probe below is separate. Successful-request percentiles exclude failures; all-attempt distributions/statuses remain in JSON/CSV. Only 15/14 successes support the 8/16-client tails, so these percentiles are unstable descriptive values.

Higher client counts mostly caused rejection. Faster high-concurrency batch completion and conditional latency do not establish increased inference capacity: admitted method/query mixes change when 85-86% are rejected, and the CPU host was not isolated (regression checks/IDE shared the machine during portions of the pass). One pass per level does not control runtime warm state, scheduling, background activity or timing variance. The current configuration did not deliver error-free service beyond the sampled one-client workload.

### Memory observations

Native `GetProcessMemoryInfo` working set/RSS of the verified serving PID, 100ms samples plus stage markers:

| Stage | Working set MiB | Private commit MiB |
| --- | --- | --- |
| before_model_loading | 61.074 | 658.426 |
| after_model_initialization | 711.367 | 2084.098 |
| after_warmup | 872.348 | 2259.043 |
| after_requests_complete | 1191.965 | 2628.930 |

Sampled peak during load: **1170.973MiB**; sampled overall peak **1234.086MiB**; cumulative OS peak **1234.148MiB**. No sampler error occurred. Private commit is not RSS. After-initialization includes earlier first-use dense/router requests but precedes configured warmup; after-requests is after the measured workloads and correctness replay, before cancellation/shutdown probes. Sampled peaks may miss brief spikes. System-total RAM was not reported because its WMI query was denied. CPU/platform/library metadata and environment freeze are saved.

Working set grew from about 872MiB after warmup to 1,192MiB after requests. This is an unresolved resource-stability observation; this pass cannot distinguish allocator retention/caching from a leak. No leak-free or stable-memory claim is made. Graceful shutdown released model references and exited; that does not establish steady-state memory boundedness for a long-lived process. A longer fixed-query soak/profile with an explicit memory budget is the next useful validation.

### Lifecycle, correctness, timeout and shutdown

Encoder, cross-encoder, BM25 index, dense cache, router/features/policy each recorded exactly **one attempt and one successful initialization**, confirmed independently by logs. Cold health loaded none; concurrent first-use did not duplicate initialization. Router used the unchanged canonical policy and left the cross-encoder unloaded until explicitly initialized/needed.

**360 direct/API correctness checks passed**: all 40 query/method pairs before load, 280 successful measured requests during load, and all 40 again after. IDs/ranks match exactly; absolute score tolerance <=1e-5, relative zero. Maximum observed successful-load score difference: 0.0. All eight queries produced different BM25/dense rankings, including public cases with previously documented relevant-document rank differences; no qrels or relevance labels were used to schedule requests or route methods. All old retrieval/data/evaluation implementation and artifact hashes remain protected.

A separate real cross-encoder call triggered HTTPX `ReadTimeout` with active=1 afterwards; a second request was explicitly task-cancelled after health observed active inference, with `CancelledError` and active=1 afterwards. CPU computation continued under the serialization lock; draining and an identical reranked recovery response were verified. This is client disconnect/cancellation recovery, not immediate server inference cancellation. Slow/throwing scorer tests over real loopback HTTP additionally verify safe 503s, capacity release and subsequent unchanged rankings.

Graceful Uvicorn shutdown during a real active reranked request returned HTTP 200 with unchanged ranks/scores, logged resource release/service stop and exited 0. The manager shutdown-admission race was repaired: stopping is signalled before waiting on the busy lock, new/queued searches fail safely with 503, and active work keeps its models until completion. Failed initialization remains sticky, with explicit attempt/success counters; a dense dependency-failure message was generalized to avoid incorrectly asserting a cache-only failure.

### Verification, artifacts and failed runs

Final suite: **173 passed in 9.49s**, including 159 existing plus 14 new tests. Ruff lint passed; **62 files already formatted**; pip check found no broken requirements. API fault integration tests use authored data/model doubles and no downloads. Final Phase 2/3/4 replays verified **1,200 / 300 / 1,615** rows. All **181** original audit-protected artifact/model/index/retrieval/evaluation/data files remained byte-identical; the load run independently protected 151 prior artifact/model/index files. Current source hashes match the measured final run.

All seven required outputs are present and nonempty, alongside raw request/hit records, direct rankings, workload, health snapshots, logs, source/protected/input hashes, exact command, environment freeze and final checks. [Latency plot](../experiments/results/phase6/phase6-20261109T153436Z-0e8e7412820d/latency_vs_concurrency.png), [working-set plot](../experiments/results/phase6/phase6-20261109T153436Z-0e8e7412820d/memory_usage.png), and [throughput/rejection context](../experiments/results/phase6/phase6-20261109T153436Z-0e8e7412820d/throughput_and_rejections.png) are data-derived; the required two plots were visually inspected. Prior Phase 1-5 outputs were preserved.

Four incomplete/invalid Phase 6 runs remain preserved:

- `phase6-20261109T150740Z-fc24027bd07c`: cold client/drain deadline failed; encoder eventually completed after ~493s and server exited gracefully. This delay came from a problematic benchmark stdin-control prototype, not evidence of normal model load performance.
- `phase6-20261109T151804Z-00f856221040`: buffered-stdin prototype deliberately stopped for diagnosis; ended with connection error.
- `phase6-20261109T152612Z-808155d5ca11`: raw CRT stdin-reading prototype still stalled native import; deliberately stopped, connection error retained. A separate diagnostic reproduced native NumPy import waiting while stdin was read and completion after releasing that reader; it does not diagnose NumPy internals universally. Private atomic control files removed the blocking reader.
- `phase6-20261109T152911Z-02e5fae500b8`: completed timing/correctness pass (262 success, 238 HTTP 429), subsequently marked **failed** after review found its RSS sampler watched the Windows venv launcher. Its raw timing/memory data and explicit invalidation remain; the ~4MiB values must not represent API memory. Final rerun uses the PID from Uvicorn's startup log and refuses to guess.

An intermediate full test run had 165 passed and six fixture setup errors after fixture-import refactoring; fixture registration was repaired. Early formatting caught a list-comprehension bracket typo and fixture lint issues; all final checks passed. No failed requests or prototype runs were discarded to improve reported results.

No immediate CPU cancellation, long-duration memory guarantee, authentication, dashboard, Docker/cloud, new router or production-readiness claim is included. Exact reproduction commands and artifact definitions are in [API reliability methodology](api_reliability.md). Recommended next step: choose an explicit request/error/memory budget and run a longer fixed-concurrency soak/profile before increasing traffic or deployment scope.


## Phase 6.1: sustained memory and reliability investigation (2026-11-09)

The registered campaign **completed** in one serving process: [exact portable report](../experiments/reports/phase6_1-20261109T162840Z-f2dea9f54ecb.json), raw `experiments/results/phase6_1/phase6_1-20261109T162840Z-f2dea9f54ecb/`. Completion does **not** mean budgets passed. Duration: 900s concurrency1, 300s drained idle, 900s concurrency2, 300s idle, 300s intentional concurrency4 stress, 300s idle, plus initialization, 60s warmup, diagnostics and fresh correctness workloads. The recorded campaign ran from 2026-11-09T16:28:40.387894+00:00 through 2026-11-09T17:22:17.749425+00:00 (UTC), about 54 minutes including setup/checks. Submission deadlines were respected; final in-flight calls drained, with actual batch durations below.

Default same eight-query public/authored pool and seed42, same fixed cycle, all five methods with equal weights: 20% attempted requests per whole cycle. Duration cutoffs can truncate a cycle; the actual per-method distributions are saved in each summary. Reranking is bounded by count but naturally dominates CPU time. No qrels were used for scheduling or routing. CPU model settings, dense cache, corpus and frozen policy are unchanged. Trace instrumentation starts after model loading and changes costs; timings are not directly comparable with untraced Phase6 runs. Regression verification finished before the first timed stage of the final campaign; IDE/host activity and OS scheduling were not isolated.

### Request reliability: normal versus stress

| Stage | Submitted seconds / drained batch seconds | Requests | Success | HTTP429 | Success % | Rejection % | Successful/s | Attempted/s | Successful mean ms | p50 ms | p95 ms | p99 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soak_1_c1 | 900 / 902.055 | 1653 | 1653 | 0 | 100.000 | 0.000 | 1.832 | 1.832 | 545.510 | 70.662 | 2919.742 | 6883.793 |
| soak_2_c2 | 900 / 905.105 | 1586 | 1307 | 279 | 82.409 | 17.591 | 1.444 | 1.752 | 947.799 | 295.668 | 3459.848 | 4578.946 |
| stress_c4 | 300 / 300.409 | 21878 | 309 | 21569 | 1.412 | 98.588 | 1.029 | 72.827 | 1648.089 | 743.431 | 5754.006 | 6418.564 |

For all three measured stages: **zero other HTTP errors, zero timeouts, zero client cancellations**. Rejections are failures in reliability statistics; they are not included in successful latency. Full all-attempt distributions remain in JSON/CSV. Stress rejection was permitted intentionally, not declared good serving capacity. Accepted method mixes differ under rejection, so throughput/latency comparisons are descriptive, not scaling claims. No immediate server CPU interruption was added or claimed; Phase6 cancellation/recovery regression tests still pass, and these steady soaks did not deliberately inject cancellation.

### Memory measurements and growth

Serving PID **20388**, Windows launcher PID **21548**; all allocation snapshots independently report the same serving PID. Sampling interval1s, native working set/RSS/OS peak/private commit and native available system memory. No sampler errors. No forced GC, allocator trim or cache clearing.

| Baseline marker | Working set MiB |
| --- | --- |
| before_initialization | 60.926 |
| after_initialization | 626.754 |
| after_warmup | 924.578 |

Sampled overall peak **1081.695MiB**; cumulative OS peak **1085.879MiB**. Total system RAM **15.676GiB**, minimum observed available memory **0.843GiB**, obtained through GlobalMemoryStatusEx (no WMI needed). A sampled peak can miss transients. Private commit is not RSS or total resident tensor memory.

| Stage | Early mean MiB | Late mean MiB | Whole slope MiB/min | Last120s slope MiB/min | Peak MiB | Workload end MiB | Idle start MiB | Idle end MiB | Actual idle change MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soak_1_c1 | 969.763 | 988.351 | 0.899 | 4.825 | 1013.312 | 970.438 | 980.484 | 980.484 | 0.000 |
| soak_2_c2 | 998.134 | 967.359 | -4.174 | -3.628 | 1081.695 | 947.855 | 958.285 | 958.285 | 0.000 |
| stress_c4 | 998.969 | 964.102 | -9.533 | -0.405 | 1045.223 | 896.484 | 898.344 | 888.664 | -9.680 |

The primary `memory_analysis.json` field `idle_change_mib` compares the last workload sample with idle end. A stopped diagnostic snapshot occurs between those points and itself allocates memory. **It must not be mistaken for growth during idle.** `allocation_interpretation.json` and the portable report add actual idle-start-to-idle-end changes from the CSV: 0MiB for both normal idle periods and -9.680MiB for stress idle. The roughly +10MiB from normal workload end to idle end includes that intervening diagnostic; there is no observed increase within the normal idle windows themselves. Idle spans were each about300s. No return to pre-model or warmup baseline was observed.

By the registered slope rule, C1 is increasing in the last120s (4.825MiB/min), C2 decreasing (-3.628), stress approximately flat late (-0.405). C1 whole-stage slope is only0.899MiB/min, with oscillating working-set samples rather than monotonic growth. The earlier interrupted C1 trial had a negative late slope (~-3.923MiB/min), underlining window/working-set sensitivity, not justifying retroactively relaxing a budget. **A blanket memory-stability pass is not established.** None of these slopes establishes a confirmed leak.

### Allocation evidence and attribution

Logical parameter bytes are {'encoder': {'parameter_bytes': 90852864, 'buffer_bytes': 8192}, 'scorer': {'parameter_bytes': 90854404, 'buffer_bytes': 8192}}; dense vector storage is **7961088 bytes (~7.592MiB). These values and all six observed `torch.Tensor` objects remained unchanged at all seven snapshots. They are lower-level logical storage counts, not the full RSS of model runtimes/imports.

Post-initialization traced current allocations rose from **443,878 bytes** to **1,178,353 bytes**, a change of **734,475 bytes (~0.700MiB)** over the campaign. Traced peak reached about4.46MiB, including diagnostic temporaries; tracer metadata overhead rose from605,568 to1,150,016 bytes. Python allocations before tracing are absent from this accounting, and most native PyTorch/library workspaces are not attributed. The large RSS-versus-traced difference must not all be asserted native memory: pre-trace Python/import memory also exists.

Total tracked-object counts were389,996 initially and387,630 after final idle. Python threads progressed4/5/7 with concurrency1/2/4; resident async task classes stayed unchanged. Some request-related objects did remain after errors: Request/JSONResponse counts were0 initially,16 after C2/idle and20 after stress/idle. These are **potential retained application objects**, not proof of an unbounded leak or proof of no retention. Their count did not grow proportionally to tens of thousands of rejections.

An isolated authored-BM25 real-HTTP probe recorded {'baseline': 0, 'after_50_rejections': 3, 'after_5s_idle': 3, 'after_100_normal_requests': 0}. It inspected frames/closure/dictionary references after rejected requests; the requests disappeared during100 ordinary successful calls without forced collection or synthetic allocation pressure. This supports collectible temporary retention in that probe. It does not prove the exact ownership/root of every object in the real-model process. The qualitative probe and verified process return code are in `experiments/results/phase6_1/audit/object_reference_probe_verified.json` and `object_probe_execution.json`; its source/hash/command are recorded separately from timed soaks.

Native allocator/workspace retention, demand paging, thread-local state and diagnostic observer effects remain plausible but not fully attributed. Late private-commit slopes were soak_1_c1: -0.173MiB/min, soak_2_c2: -4.618MiB/min, stress_c4: 5.218MiB/min. They do not track RSS slopes identically. No native allocation profiler or full tensor-lifetime trace was performed, and no product memory-management change was justified or implemented.

### Budgets, correctness and lifecycle

Pre-registered local targets: peak RSS<=2048MiB, late RSS growth<=2MiB/min, normal rejection<=5%, stress rejection<=100% intentionally, zero timeouts/cancellations/other errors, successful p95<=5000ms. All three stages **completed but failed at least one budget**:

- C1: late RSS slope exceeded2MiB/min. Reliability/p95/peak targets passed.
- C2: rejection17.591% exceeded5%. Memory/p95/other-error targets passed.
- Stress: successful p95 5754.006ms exceeded5000ms. Its rejection98.588% was permitted as intentional overload, not support for serving that load.

The second normal soak and stress reused the same serving PID, models and indexes. Every resource had one initialization attempt and one success at all three health checkpoints. Frozen policy identity/strategy and ordered IDs/ranks/scores were checked against direct retrieval: **3621 checks passed**, including warmup and all successful live calls plus160 fresh initial/post-idle query/method checks. Absolute score tolerance1e-5, relative zero; no during-workload mismatch. Graceful final shutdown logged resource release/service stop and exited0. No server remains from the final campaign.

### Failures, fixes, preserved outputs and final checks

- First smoke `phase6_1-20261109T155639Z-0e69a8ef4747` failed on a native class module descriptor during object census. Safe type labeling and a regression test fixed it. The next functional smoke `phase6_1-20261109T155820Z-d562849a95c4` completed20s C1/20s C2/10s stress with5s idle each; those short windows are not long-duration stability evidence.
- First long campaign `phase6_1-20261109T160233Z-7529df6e4a53` completed900s C1 (1,586 successes, no rejections) and its300s idle but failed before post-idle diagnostics: a transient Windows PermissionError reading the atomic control file killed the controller thread. The owned helper was stopped, partial CSV/report/logs retained, and `partial_memory_analysis.json` labels a post-hoc analysis of the completed portion only. It is not counted as a completed campaign or graceful shutdown.
- The benchmark controller now retries transient reads, publishes replies atomically, and the parent retries transient/partial reply reads. Fault injection verifies a denied first read recovers. The final full campaign completed the formerly failing checkpoint and all remaining stages. Its actual transient-read retry counter was 0; the regression is controlled evidence of retry behavior, not a claim a denial occurred in this clean pass. These changes affect benchmark I/O only, not serving retrieval behavior.
- The first reference-probe PowerShell stderr-redirection invocation reported a shell error even though its JSON completed. A structured subprocess rerun returned0 with verified statuses/shutdown. An unused diagnostic-script import caught by Ruff was removed.

Final full suite **185 passed in18.66s**; Ruff lint passed, **67 files already formatted**, dependencies passed. Final replays: Phase2 **1,200**, Phase3 **300**, Phase4 **1,615** rows. All **288** original audit-protected prior artifact/model/index/application files remained byte-identical. Recorded timed-source hashes match current files; the later qualitative probe has its own supplemental hash. Separate `docs/environment-phase6_1-freeze.txt` preserves the actual environment without replacing earlier freezes.

All eight required artifacts, allocation diagnostics, streamed request outcomes, workload/reference/source/hardware/configuration/command metadata, final checks and four data-derived figures exist. The [memory](../experiments/results/phase6_1/phase6_1-20261109T162840Z-f2dea9f54ecb/memory_vs_time.png), [latency](../experiments/results/phase6_1/phase6_1-20261109T162840Z-f2dea9f54ecb/latency_vs_time.png), [method outcomes](../experiments/results/phase6_1/phase6_1-20261109T162840Z-f2dea9f54ecb/outcomes_by_method.png), and [idle recovery](../experiments/results/phase6_1/phase6_1-20261109T162840Z-f2dea9f54ecb/idle_recovery.png) figures were visually inspected. Normal and stress statistics must remain separate; the combined outcome plot is dominated by intentional stress rejections.

**Containerization readiness is not established under these declared normal-operation budgets.** No confirmed leak was found, but native allocation attribution and the failed C1 growth/C2 rejection targets remain unresolved. A packaging-only local container prototype is a separate upcoming milestone, not proof of stable multi-client service; memory/cgroup limits and behavior must be revalidated on the target OS. Recommended next investigation: native/private-commit profiling and preregistered repeated same-concurrency cycles/window sensitivity, plus an explicit admission/latency target. Do not force GC or tune against these results to manufacture a pass. No Docker, deployment, new product feature, retrieval algorithm, routing policy or fine-tuning was added. [Exact commands and diagnostic methodology](soak_testing.md).


## Phase 6.2: native memory and repeatability (2026-11-10 local)

Five independent serving runs completed on the existing Windows CPU environment. Every run retained the Phase6.1 eight-query/equal-five-method seeded cycle,60s warmup,1s RSS/private sampling,30s timeout and300s idle. C1/C2 workloads each submitted for900s; C4 stress for300s. Durations include a separately recorded final request drain. All five serving processes exited0 with resource initialization counts1/1. This campaign finished2026-11-09T21:25:37UTC, before the later interrupted diagnostic control.

[Machine-readable report](../experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/phase6_2_report.json), [per-run comparison](../experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/repeated_run_comparison.csv), [all request outcomes](../experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/request_outcomes.csv), and [exact PowerShell commands/methodology](native_memory_profiling.md).

### Measured outcomes

| Run | Attempts | Successes |429|429/all attempts|Successful requests/s|
|---|---:|---:|---:|---:|---:|
|c1_r1|1492|1492|0|0.000%|1.654|
|c2_r1|1924|1600|324|16.840%|1.777|
|c1_r2|1711|1711|0|0.000%|1.899|
|c2_r2|1693|1399|294|17.366%|1.549|
|c4_stress|23323|303|23020|98.701%|1.006|

Measured workloads total30,143attempts/6,505successes/23,638rejections. Request timeouts,cancellations and other request errors were0 in every run. Warmup is excluded from these statistics but preserved as a separate stage in outcomes. All-attempt and per-method rates/latency distributions, actual elapsed time and drain overrun remain in the JSON. Fast429responses are not included in successful-request latency.

| Run |Successful mean ms|p50 ms|p95 ms|p99 ms|429 mean ms|429 p95 ms|
|---|---:|---:|---:|---:|---:|---:|
|c1_r1|604.383|123.137|2779.231|3465.002|n/a|n/a|
|c2_r1|715.816|214.321|2703.566|2969.350|2020.663|2040.553|
|c1_r2|526.449|119.990|2521.152|2641.323|n/a|n/a|
|c2_r2|861.586|296.256|3131.706|4020.154|2033.440|2074.397|
|c4_stress|1703.090|734.011|5551.193|6391.772|29.662|36.254|

### Memory, variability and budgets

All memory values use MiB (2^20bytes). Peaks below are workload-only; overall sampled maximum was1130.656MiB and observed OS lifetime peak1131.219MiB, including checks/setup. Private commit is not resident RSS.

|Run|Peak RSS|Peak private commit|Whole RSS MiB/min|Late RSS MiB/min|Late private MiB/min|End idle RSS|End idle private|
|---|---:|---:|---:|---:|---:|---:|---:|
|c1_r1|1022.062|2361.688|4.809|8.440|8.237|985.594|2337.000|
|c2_r1|1071.621|2450.535|5.866|4.037|4.907|1039.914|2416.621|
|c1_r2|1023.680|2362.168|4.110|4.261|2.114|995.125|2338.613|
|c2_r2|1054.168|2422.648|2.699|7.453|0.833|1012.473|2384.539|
|c4_stress|1103.797|2511.723|4.970|-12.542|-16.035|1055.980|2466.629|

Actual idle-start/end RSS changed0/0/-0.00390625/0/0MiB in run order; private commit changed0 in all five. Workload-end to idle-end includes intervening diagnostic allocation and is not the pure idle recovery metric. End-of-trace drops during resource release/shutdown are also not idle recovery.

Unchanged budgets: RSS<=2048MiB, late RSS growth<=2MiB/min in the final120s after at least240s observed, normal429<=5%, intentional stress429<=100%, zero timeouts/cancellations/other request errors, successfulp95<=5000ms. **All five runs failed at least one budget.** Every peak/reliability check passed. All four normal runs failed growth; both C2 runs also failed rejection. C4 passed its deliberately permissive stress rejection and late-growth checks, but failed successfulp95 at5551.193ms. Stress rejection98.701% is not multi-user capacity evidence. Private-commit trends are observations, not a newly invented pass threshold.

Two-run descriptive variability: C1p95 ranged2521.152?2779.231ms (sampleSD182.489ms), throughput1.654?1.899successes/s, late RSS4.261?8.440MiB/min. C2p952703.566?3131.706ms (SD302.740), throughput1.549?1.777/s, rejection16.840?17.366%, late RSS4.037?7.453. Two repetitions do not establish a distribution/confidence interval. For example,C2repeat2 late RSS+7.453 versus private+0.833MiB/min illustrates their different behavior. C4whole RSS+4.970 despite late-12.542 likewise does not establish lifetime boundedness.

### Native findings and controlled probes

Actual inference workers reported PyTorch intraop4/interop14, while two OpenBLAS runtimes were configured for20threads each; multiple OpenMP runtimes had distinct settings. Do not add configuration limits to infer active concurrency. Native checkpoint counts reached72/75/72/75/83 by run; these are observed checkpoint counts, not guaranteed transient thread peaks. Logical model/index sizes and torch.Tensor census remained unchanged. Traced Python and tracer overhead were measured separately and cannot be subtracted from RSS/commit to label the remainder a native leak. Full observations and controlled-probe sources are in memory_attribution.json.

The authored NumPy import/kernel control changed only child OPENBLAS_NUM_THREADS1vs20. Post-import private commit17.164vs627.773MiB; after100 identical normalized float32 matrix-vector calls58.910vs669.457MiB. RSS remained roughly26?39MiB. Scores differed by at most2.98e-08 with identical ordering. This identifies a substantial thread-configuration contribution to private commit; it is not an IR quality benchmark or an adopted service setting.

Fresh fixed-query inference controls initialized every resource identically, warmed5calls, repeated40calls, then profiled one separate CPU call. Primary controls are in inference_controls_clean; only pipeline choice differs for hybrid versus reranked, and only child OPENBLAS_NUM_THREADS differs for reranked versus reranked_blas1. All repetition ranking/score checks passed. The following snapshots precede the separate profiler:

|Condition|After warmup RSS|After40 RSS|After warmup private|After40 private|Private change|
|---|---:|---:|---:|---:|---:|
|hybrid|658.891|658.980|2031.516|2031.516|0.000|
|reranked|826.160|816.875|2219.379|2208.262|-11.117|
|reranked_blas1|823.293|812.711|997.203|984.812|-12.391|

Default versus BLAS1 reranked private commit differed1223.449MiB after40calls while RSS differed only4.164MiB. Ordered IDs/ranks matched: True; maximum score delta0 (unchanged absolute tolerance1e-5). Runtime configurations confirm both OpenBLAS pools20vs1. No serving configuration was changed.

All three CPU operator profiles completed. Reported self CPU memory for aten::addmm was1660416bytes for the hybrid search and851037776bytes for reranked (same reranked value under BLAS1). These are aggregated framework-reported operator allocation counters in one profiled search, not one durable buffer or retained RSS; inclusive operator values overlap. Profiling creates its own allocations, and allocations existing before its scope cannot be fully tracked. These controls support native-pool/private-commit and reranking-temporary contributors; they do not attribute every anonymous heap allocation or explain all varied-query/HTTP retention.

**Leak status: Inconclusive.** No allocation-stack/heap-owner trace was captured. WPR profiles were discoverable, but no ETW heap/VirtualAlloc recording or global heap-tracing settings were applied. Windows virtual-region types and CPU tensor allocation counters do not prove allocation ownership or absence of a leak. Repeated positive normal-run growth remains unresolved. No forced GC, allocator release or speculative production optimization was applied.

### Diagnostic issues and verification

The initial short smoke completed7/7measured successes and graceful shutdown, but p956014.560ms failed the latency target; its10s duration was inconclusive for stability. Review found continuous native instrumentation patched an imported canonical module rather than the executing __main__: all five long runs have genuine1s RSS/private samples and stopped native checkpoints, not1s native-thread/region traces. Their sparse native CSV is labeled explicitly; no interpolation was invented. The original and timed harness sources are archived. The module-wiring fix has a regression test; its real short verification is pending below. Production service/retriever sources were untouched.

C4reported one sampler OSError. No timestamp was recorded for that error; polling after process exit is a plausible explanation, not established attribution. Largest observed intersample gaps across the five runs were1.098/1.060/1.041/1.156/1.106s, with full workload and idle coverage. Preserve this error separately from the zero request errors.

The first inference-control attempt recorded30014.849s before model initialization completed, spanning an overnight interruption. Its raw outputs/validity note are retained and the entire attempt is excluded from the primary comparison. All conditions were rerun fresh; clean model initialization took11.4?12.6s. All five serving runs had already finished before that interruption. A fixture also reproduced/fixed missing-slope variability handling; temporary helper lint issues were corrected.

Final verification: **193tests passed in20.09s**; Ruff lint passed,77files formatted, dependencies passed. Replays verified Phase2:1200rows, Phase3:300, Phase4:1615. All7416API/direct ranking/score/policy checks passed and all five serving processes exited0. All423audit entries are accounted for:422byte-identical, with only the intentional api_soak benchmark extension changed; no unexpected protected-file changes. Timed runtime source hashes match across all five runs and their archived copies, and the frozen policy hash is identical.

The corrected real smoke `phase6_2-20261110T055917Z-deed7c48cf51` completed15/15measured successes,99total correctness checks, exit0 and no sampler errors. All97memory rows now contain native thread and virtual-region fields, versus0native rows before the fix. The ten-second growth result remains Inconclusive. Native observation cost averaged198.243ms (max484.663ms); the configured1s interval is followed by observation time, so this is not exact1Hz and can perturb workloads. The five published long runs used the earlier inexpensive RSS/private sampler plus native checkpoints; do not retroactively attribute that new observer cost or continuous trace to them. No product memory-management fix was made.

[Final verification/coverage/source evidence](../experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/final_validation.json) and [portable report](../experiments/reports/phase6_2-repeat-20261109T194018Z-c8b6ba59.json) preserve actual commands/results. All eight required artifacts and five figures exist. RSS/commit, latency and idle figures were visually checked. Remaining limitations include sparse native observations in the long runs, one untimestamped terminal sampler error, host/observer effects, two normal repetitions/one stress repetition, and no native heap-allocation ownership trace.

### Deployment decision

|Scope|Recommendation and limit|
|---|---|
|Local Docker testing|Reasonable next isolated packaging experiment, not validated container readiness. Re-run these budgets/resource observations on the target OS/cgroup; Docker remains unimplemented.|
|Bounded single-user deployment|Stable deployment under the declared growth budget is not justified; monitored, time-bounded local research/demo use is the supported interpretation.|
|Small multi-user deployment|Not recommended under the current5% rejection target; both C2 repetitions failed it.|
|Larger production deployment|Not established; growth/admission/latency failures, limited workload, no host isolation and incomplete native attribution remain.|

[RSS versus private commit](../experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/rss_vs_commit.png), [independent-run RSS](../experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/rss_across_runs.png), [idle recovery](../experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/idle_recovery.png), [successful latency](../experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/latency_vs_concurrency.png), and [rejection by method/run](../experiments/results/phase6_2/repeat-20261109T194018Z-c8b6ba59/rejection_by_method.png) derive from saved observations. No Docker, MLflow, cloud, new retrieval/routing algorithm, API feature or fine-tuning was added.
## Phase 7A: local Docker validation (2026-11-10)

Completed campaign: `experiments/results/phase7a/phase7a-20261110T063328Z-3ae104cc`; [portable report](../experiments/reports/phase7a-20261110T063328Z-3ae104cc.json). [Container guide](docker_deployment.md) provides exact PowerShell commands, configuration, persistent paths and methodology. This is local packaging validation, not deployment readiness or a resolution of Phase6.2's failed budgets.

One pinned Python3.11.14 Bookworm CPU image, non-root UID10001, exec-form single-worker API, existing health endpoint,60s SIGTERM grace, read-only rootfs and four read-only binds: dataset, snapshots, separate Linux index and frozen capsule. No model weights, datasets, experiments, venv or Git history are in the image. Actual image audit found64allowed files/344655bytes under the application working directory; all source hashes match the host. Default BLAS configuration is unchanged. The explicit BLAS1 condition uses a separate Compose override only.

Docker client/server28.0.4, Compose2.34.0, Linux WSL2 kernel6.18.54.1,20logical CPUs, Docker VM8151511040bytes RAM. Local Python3.11.5 Windows and container Python3.11.14 Linux differ; the CPU torch wheel is2.9.0+cpu versus native2.9.0. Linux runtime/transitive pins and actual installed freeze are retained; this is not an isolated measurement of Docker overhead alone. No disk cache flush or CPU isolation was performed. Direct inference exited before HTTP workloads; final regression checks ran after the measured campaign and lifecycle probe.

### Build and preparation

Image `rankforge:phase7a`, ID/digest `sha256:d21b8012819d47f0669f8d1764c8c8cf7a2fcad903004329586158846b787e84`. Docker-reported image size **412571272bytes (393.46MiB)**. Successful measured build **134.518s**, with BuildKit cache allowed; base layers were already cached from the first attempt. First full context transfer347.36kB; the retry transferred5.08kB incrementally, not a claim that the complete context was5kB.

The first build failed `pip check`: Linux Hugging Face Hub requires hf-xet. Added explicit hf-xet1.2.0; retry build/dependency check passed. [Initial failed-attempt record](../experiments/reports/phase7a_initial_build_failure.json) is preserved. Docker Desktop initially needed starting; sandbox engine access required permission. No certificate validation was disabled.

The original Windows cache correctly cannot represent the Linux CPU library identity. Existing offline indexing CLI built `indexes/scifact-minilm-linux.npz`,5183documents x384dimensions; embedding-build timer **265.406s**. Preparation was **277.695s** end to end (Compose startup/import/hash/index/I/O included). Windows and Linux caches are separate, and no identity check was bypassed. The corpus, both snapshots, capsule and all existing indexes were retained unchanged during serving/restart (55resource files verified).

### Startup, initialization and warmup

| Condition | Health seconds | First five pipelines seconds | Initialization component seconds | Explicit warmup seconds |
| --- | ---: | ---: | ---: | ---: |
| Native Windows |1.495|21.799|18.423|2.991|
|4GiB /4CPU default |3.530|17.642|13.426|2.919|
| Same container restarted |2.277|12.196|8.528|2.637|
|2GiB /2CPU default |3.064|15.157|8.882|5.965|
| Same container restarted |1.791|14.485|8.530|6.026|
|4GiB /4CPU BLAS1 experiment |1.550|8.328|6.710|1.339|
| Same experimental container restarted |1.150|7.147|5.615|1.049|

Initial container health includes Compose create/start; restart health timing begins after the Compose start command returns, so those columns have different startup boundaries. First-use time includes sequential complete HTTP requests and inference, not just model loading. Initialization is the sum of nonoverlapping per-request initialization increments over the first five calls, not recursively summed resource durations. All-resource states/durations and counters remain in startup JSON. Health is model-free liveness; all-method requests established readiness separately. Process startup was cold, filesystem caches were not guaranteed cold.

### API latency and correctness

Same Phase6eight public/authored queries, equal-five-method seed42schedule, k10,40attempts per C1/C2 level for each of seven native/container/restart passes.40reference pairs and five `/compare` results per pass were checked against the direct native baseline, plus every successful measured response. No qrels or effectiveness scores were used.

Successful C1 mean HTTP latency in milliseconds, eight samples per method per condition:

| Method | Native |4GiB/4CPU default |2GiB/2CPU default |4GiB/4CPU BLAS1 experiment |
| --- | ---: | ---: | ---: | ---: |
|BM25|32.681|50.237|70.369|13.589|
|Dense|44.654|159.561|204.935|12.305|
|Hybrid|74.245|200.546|292.925|22.923|
|Reranked|2893.397|2257.536|5119.847|1021.620|
|Frozen router|88.438|196.186|281.847|27.056|

These small descriptive samples do not establish performance equivalence, significance or a production BLAS recommendation. Complete means/linear p50/p95/p99, restart and C2 results are in the raw report. Largest first-container compatibility score difference was2.8014183e-6; ordered IDs/ranks matched exactly and all scores passed the unchanged absolute1e-5 tolerance. Frozen policy identity stayed `31b0cef1d5969d2b79953be683050877038c34fabc4c560b4a1b8be8f90b67f8`.

**840correctness checks passed,0failed. 560measured attempts:525successful,35HTTP429,0timeouts/transport/cancellation/other errors.** Reference checks and first-use/warmup are outside those560attempts. All C1requests succeeded. C2rejection fractions: native15%; default4CPU12.5% (restart17.5%); default2CPU25% (restart17.5%); explicit BLAS1zero in both short passes. The5%normal rejection target is therefore not met by default cases. C2reranked p95at2CPU was5873ms and5013ms after restart, also exceeding the historical5000ms target. Short passes cannot classify the historical late-growth budget.

### Resources and persistence

| Condition | Sampled process peak RSS MiB | Largest observed cgroup lifetime peak MiB | Mean cgroup CPU%, one core=100 | Success /429 |
| --- | ---: | ---: | ---: | ---: |
|Native|975.289|unavailable|unavailable|74/6|
|4GiB/4CPU default|1182.809|1715.309|336.34|75/5|
|Restart|1079.082|1072.547|356.20|73/7|
|2GiB/2CPU default|1167.406|1112.363|190.95|70/10|
|Restart|1053.285|1045.512|190.98|73/7|
|4GiB/4CPU BLAS1 experiment|1204.125|1207.617|322.49|80/0|
|Restart|1134.406|1132.043|331.48|80/0|

Each row contains80attempts, combining C1/C2. Configured cgroup ceilings were4096or2048MiB; CPU quotas were400000/100000or200000/100000. OS lifetime process RSS peaks exceeded sampled peaks:1276.402/1172.453,1213.816/1146.926,1297.926/1223.070MiB for initial/restarted containers. Sampling misses short peaks. Process RSS and cgroup charged memory are distinct; shared page charging and sequential snapshots can yield RSS above the current cgroup total. The [kernel memory ownership documentation](https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory-ownership) explains the accounting distinction; exact page ownership was not profiled. Do not infer a leak or allocator saving by subtracting these values. Windows private commit is recorded only for the local process; the VM serving PID is not a Windows PID. Container limits are not host memory measurements.

All three initial containers were healthy before stop. All six initial/restart stops exited0; no OOMKilled or automatic restarts, and sampled cgroup OOM/max events stayed0. All seven passes began with zero initialization counts and ended with exactly one attempt/success for all ten resources. Restart repeats in-memory model/index initialization as expected, reusing unchanged disk files. All samplers reported zero errors.

The separate [active-shutdown/recreation probe](../experiments/results/phase7a/rf-lifecycle-c222aa2d/lifecycle.json) observed active reranked inference before SIGTERM, received HTTP200, observed exit0 and resource-release logs, then recreated a different container. The router's rankings/policy matched, persistent resources were available and the scorer remained lazily unloaded after router-only initialization. No test containers remain running. Stopped-container Docker health text is not treated as a running-health failure; its actual health command exit was0. Missing required resources exited1before API startup, without downloading anything.

No lower Linux peak RSS was established by the BLAS experiment: its sampled peak was higher than the initial default case. Cgroup peaks are confounded by page-cache ownership/order. Observed latency/rejection improvements justify further controlled testing, not silently adopting the Phase6.2 setting or transplanting Windows private-commit savings to Linux.

### Artifacts, checks and next milestone

All seven required JSON/CSV artifacts exist, plus image/configuration/version/build/server logs, full rankings/request results, hashes, derived measurements and three plots: [latency](../experiments/results/phase7a/phase7a-20261110T063328Z-3ae104cc/api_latency.png), [process RSS](../experiments/results/phase7a/phase7a-20261110T063328Z-3ae104cc/memory_usage.png), [startup/first use](../experiments/results/phase7a/phase7a-20261110T063328Z-3ae104cc/startup_initialization.png). Raw observations are preserved; the summarizer accepts the first runner's dictionary-text CSV counters as well as current JSON counters. Later verifier changes improve CSV encoding, failure finalization and explicit experiment-environment isolation; they do not alter saved measurements or serving code.

Final regression suite: **204tests passed in16.52s**. Dependency checks passed locally and inside the image; Phase2/3/4replay verified1200/300/1615rows. Initial final lint found an unused import/formatting in a newly written verification helper; both were fixed, with final lint/format outcomes recorded separately. Earlier audit423entries:417byte-identical; five intentional API/config/logging source changes in this phase and the already documented Phase6.2api_soak change account for the rest. No protected earlier result/model/index/policy/retrieval/metric file changed.

The container is suitable for the **next monitored local MLflow integration experiment** with these resource/lifecycle limits retained. Stable deployment readiness remains unestablished: default rejection/latency targets fail in these short samples, long-duration Linux growth was not measured, and the leak remains inconclusive. No cloud deployment, Kubernetes or MLflow implementation was added.
## Phase 7B: optional MLflow tracking (2026-11-10)

Final verification: `phase7b-20261110T073341Z-12c807e3`, [portable report](../experiments/reports/phase7b-20261110T073341Z-12c807e3.json), [PowerShell commands and architecture](mlflow_tracking.md). The original local experiment files remain canonical; tracking is optional and happens after persistence. No retrieval, metric, policy, API or production BLAS behavior changed.

MLflow server/client version **3.5.1**, tracking/UI URI **http://127.0.0.1:5000**. The digest-pinned official server uses the Compose `tracking` profile, one worker, persistent SQLite backend and a separate HTTP-proxied artifact volume. Both volumes survived restart and complete container recreation. Host publication is loopback only, host validation is enabled, and a request with an untrusted Host header returned403. No authentication was added for this trusted local workstation; public/multiuser deployment remains out of scope. The API has no dependency on tracking. MLflow is left healthy for UI inspection, and the API is stopped after its verification probe.

### Actual tracked/imported scope

The finalized `rankforge-tracking-v2` registrations verified here comprise six historical parent runs /124child rows, plus two new parent runs /two children. This counts the finalized registrations, not every retained development trial in the database.

| Historical source | Child rows | Metrics family |
| --- | ---: | --- |
| Phase2 SciFact, all300queries |4|Evaluation quality/offline latency/setup |
| Phase3 SciFact internal60held-out queries |5|Evaluation quality/offline latency/setup |
| Phase4 NFCorpus,323queries |5|Evaluation quality/offline latency/setup |
| Phase6.1 long operational campaign |15|Service metrics by method/stage |
| Phase6.2 five-run repeated campaign |25|Service metrics by method/run |
| Phase7A native/container/restart comparisons |70|Service metrics by method/condition/concurrency |

All six historical imports read saved artifacts without rerunning model inference. Logged parameters and metric dictionaries exactly matched the adapters' saved-source values. Repeated imports returned the same completed parent IDs. The integration verifier performed3732numeric equality comparisons, including repeat checks after recreation; the additional public run added6, for **3738numeric comparisons passed**. This is an equality-check count with repeated validations, not3738distinct scientific metrics. No missing p99/initialization/other values were invented. Missing values are omitted. `eval.*`, `ops.*` and `setup.*` namespaces separate quality, service and experiment setup measurements.

New tracked executions: a genuine authored smoke benchmark and a genuine **25-query SciFact BM25 subset**, both with source parameters/metrics verified against MLflow. The public subset used all5183documents, split test, seed42, k10; it is an already exposed plumbing/evaluation subset, not a tuning/untouched validation set. Its measured local results were:

| Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Mean first-pass retrieval ms | BM25 index seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
|0.640000|0.068000|0.551333|0.572347|15.434928|0.595572|

The public parent is `249f40c396734b5eb341a4daa5b0d70d`; child `63c0f49d1937444081bf3c2a2902bbce`. Its canonical source is `experiments/results/phase7b/phase7b-20261110T073341Z-12c807e3/new_scifact/20261110T073902Z-c5ce84baaaaa`. No old synthetic/public results were overwritten, and smoke results are not represented as public-data quality evidence.

### Persistence, independence and preservation

The source-manifest artifact SHA256 was identical before restart, after restart, after recreation with a different container ID, and after outage recovery: `3d6be4d67329f75db59ca2e443889cea153e2454174f0f46aee4b37ad0ac13fa`. All imported parameter/metric records remained readable after recreation. This verifies the tested artifact and backend persistence; it is not a backup/disaster-recovery certification.

With MLflow stopped, an enabled-tracking smoke command completed its local benchmark and wrote canonical summary/metadata/rankings; its separate receipt correctly recorded tracking failure. The real Docker API then returned correct BM25, dense, hybrid, reranked and frozen-router results for compatibility query94, matching preserved ordered IDs/ranks and the existing absolute1e-5score tolerance; the policy hash was unchanged. The verifier restored MLflow afterward.

Sixty-six historical source/artifact files were unchanged byte-for-byte during import/verification. All55Phase7A dataset/model/index/policy resource hashes also remained unchanged. Of64image-audited source files,60remain identical; the four intentional differences are post-persistence tracking hooks in benchmark/comparison/phase3/phase4CLIs. Retrieval, metrics, router/features, API and cache validation implementations remain unchanged. Uploaded artifacts are filtered JSON/CSV/plots/configuration/provenance copies; raw datasets, model weights, indexes, caches and request document payloads are excluded. Unit tests verified filtering, credential redaction, source-hash rejection and mutation detection.

### Reproduced failures and fixes

The sandboxed optional-package installation could not reach the index; the authorized network retry installed the pinned skinny client. Its packaging constraint changed the local packaging distribution from26.3to25.0; dependency checks and the complete regression suite passed afterward, and `docs/environment-phase7b-freeze.txt` records the actual environment. Inference-model identity versions stayed unchanged.

Docker's engine returned HTTP500after pulling the image, including failures for its version/ping endpoints. A Docker Desktop restart restored operation. The first real registrations exposed MLflow3.5.1's artifact repository consulting fluent URI state even when the client had an explicit URI; scoped/restored artifact-URI handling fixed it. Subsequent attempts exposed SDK run-link emoji output failing on Windowscp1252; the supported print-suppression setting fixed that. Failed partial runs and local receipts remain preserved. A mock credential test caught URI/environment-assignment redaction precedence; its fix passed. A temporary finalization-helper unused import was removed before final lint passed.

The initial v1pilot is retained separately. Before finalization, schema v2 added normalized parameter/metric identity and explicitly separated report-host Python from container Python/image/software metadata. Earlier v1trial runs are not the final evidence. Use completed v2parent tags and comparison groups in the UI; do not pool trials, incomplete parents, different datasets/scopes, HTTP versus offline timings or unequal native/container configurations.

Final checks: **218tests passed in16.69s**, Ruff lint and formatting passed, local dependency check passed, Phase2/3/4replays verified1200/300/1615rows. Fourteen new tests use client doubles/authored fixtures; ordinary tests need neither MLflow installed nor a server. Real UI verification covered HTTP HTML plus a JavaScript asset (both200); interactive browser rendering was not tested.

Remaining limits: SQLite/single-worker local service; no distributed atomic import uniqueness, concurrent importer guarantee, authentication/public hosting or backup certification. Credential filtering is bounded to known experiment artifacts, not a general secret-scanning service. Historical metrics are copied measurements, not fresh scientific results. Phase6.2/7Aresource-budget failures, unmeasured long-duration Linux stability and inconclusive leak remain unchanged. No cloud, Kubernetes, model registry, weight upload or MLflow-based serving was added.
