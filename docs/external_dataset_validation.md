# Phase 4: external NFCorpus validation

This experiment transfers the exact Phase 3 routing policy to official BEIR NFCorpus, without tuning or changing retrieval/metric definitions. NFCorpus was selected explicitly in the Phase 4 specification. It provides a different medical/nutrition retrieval task and query/judgment distribution from SciFact scientific-claim evidence retrieval. Both collections remain broadly biomedical; one additional collection cannot establish general-domain or production-scale generalization. Pretrained-model training overlap has not been audited, so external here means external to routing-policy development, not guaranteed unseen by pretrained models.

## Policy freeze and evaluation protocol

The exact Phase 3 frozen_policy.json is copied byte-for-byte before external preparation. Canonical policy-record SHA256: `31b0cef1d5969d2b79953be683050877038c34fabc4c560b4a1b8be8f90b67f8`; raw-file SHA256: `68a6b6e0251f0f1b4da471e96e30711053c332d32704a0572483cf5f49213617`. Parameters remain rare_min=1.01, rerank_max_tokens=0, coverage_min=0.8. This is hybrid-only for queries with lexical tokens; the original empty/tokenless BM25 branch is preserved, not replaced.

Registration is `experiments/configs/phase4-transfer/`: protocol.json plus frozen_policy.json. It records the source Phase 3 run, its metadata hash, exact policy bytes/canonical hash, and the unchanged router/feature-extractor implementation hashes. Registration timestamp is 2026-10-31T07:11:51.226254+00:00. Dataset preparation/first qrels parsing occurred afterward, at 07:12:48.494563+00:00. The downloader and evaluator require the frozen protocol before parsing external labels. Phase 4 never calls policy selection and never passes external outcomes to a tuning function. Existing Phase 3 policy selection remains untouched. Model identities/configuration are checked against the saved source before inference.

The official test split was selected in the registration before label inspection. Train/dev/test are inspected for format, references, counts and grades, never for optimizing thresholds. All official test query IDs and all corpus documents are retained; there is no evaluation subset, relevance-based document filtering or manual query exclusion. Exact IDs accompany run metadata.

## Dataset provenance, license and counts

Official archive: [BEIR NFCorpus ZIP](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip). The [BEIR dataset table](https://github.com/beir-cellar/beir#beers-available-datasets) publishes MD5 `a89dba18a62ef92f7d323ec890a0d38d`; import requires that digest. Verified raw archive SHA256: `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`. No semantic release number is invented: the archive digest is the version identifier.

The [BEIR NFCorpus card](https://huggingface.co/datasets/BeIR/nfcorpus) lists [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/). Preserve creator attribution, the license link and change notices, and ShareAlike terms when distributing adaptations. Original project site was queried but timed out; the BEIR listing/card and CC license terms were accessible and checked before download. Credit [NFCorpus: Boteva, Gholipour, Sokolov and Riezler, ECIR 2016](https://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf) and BEIR (Thakur et al., 2021). Raw data stays local in ignored data/public/, with no external publishing in this task.

The raw archive is preserved at data/public/archives/nfcorpus.zip. Required corpus/queries/qrels files are copied byte-for-byte, their hashes recorded, and all split references validated with the existing loader before staged import completes. Arbitrary ZIP members are never extracted. Additional archive members remain available in the preserved ZIP. Repeat imports verify existing file hashes; altered inputs require another destination rather than silent replacement. Metadata records preparation date, source, archive/file digests, policy identifier and observed counts.

Observed counts (computed after validated import, not constants in evaluation code):

| Item/split | Queries | Judgments |
| --- | --- | --- |
| All raw query IDs | 3,237 | - |
| train | 2,590 | 110,575 |
| dev | 324 | 11,385 |
| test | 323 | 12,334 |

The shared corpus has 3,633 documents. All split ID sets are disjoint, and no raw query ID lacks a judgment in all splits. Train grades are all 1; dev grades: 10,864 ones and 521 twos; test: 11,758 ones and 576 twos. No split contains a query without positive judgments. Train/dev counts are provenance checks, not tuning data for this transfer experiment.

Positive grades remain relevant for binary Recall/Precision/MRR; nDCG uses the existing exponential gain `2**grade - 1` (grade 2 therefore gains 3). Unjudged documents have zero evaluation gain as a convention, not known irrelevance. NFCorpus has substantially more positive judgments per query than SciFact. Recall@10 and indicator counts cannot be compared across collections as controlled quality differences: at most ten positives can be retrieved, and a fixed top-20 pool may omit some positives on a heavily judged query even with useful results. Grade distribution also affects nDCG. Metric definitions were not changed to improve scores.

## Pipeline and timing

All five methods run independently on the same NFCorpus test IDs/documents/qrels/cutoff: BM25, dense, hybrid RRF, hybrid+cross-encoder, frozen router. The common evaluate_live routine from Phase 3 is reused. No results are borrowed across separately timed methods. Router timing includes lexical features and decision; baselines and router use the same warmed CPU conditions, rotating the five-system execution order per query. Model loading, corpus feature statistics and index construction/validation occur before query timing. One authored-text warmup executes per complete pipeline. p50/p95 summarize one measurement per query, not repeated-run performance uncertainty.

Frozen settings: k=10, BM25 k1=1.5/b=0.75/epsilon=0.25 and unchanged tokenizer; source candidate depth 50, equal-weight RRF constant 60; rerank pool 20, batch 16, four torch threads, CPU; dense max length 256 and cross-encoder pair max length 512. Snapshots are the same pinned MiniLM models from Phases 2/3, checked by revision/file hashes and library settings. Only the dataset/index path changes to indexes/nfcorpus-minilm.npz. The separate index's original embedding-build duration is retained; cache-hit setup is not presented as fresh construction time. The reranker observer saves its existing candidate trace without changing ranking logic.

Candidate counts denote actual cross-encoder-input documents. A zero for a non-reranked path means no cross-encoder pool was requested. Configured RRF source depth is separately retained. Failure analysis uses the saved 20-document pool, not an inferred full source union. Incremental reranking time is recorded; retrieval-component latency is total minus that incremental time and is labeled a subtraction estimate.

## Analysis and uncertainty

The shared Phase 3 taxonomy retains observed candidate/ranking failures and metric regressions, with lexical/semantic/short-query indicators explicitly inferred. It is not assumed that category rates match SciFact. Category plots compare fractions, include sample counts, and flag the different judgment density/task. Complete ranked IDs/scores, per-query metrics, counts, positive positions, candidate pools, routes, evidence and pairwise nDCG changes are persisted.

Paired bootstrap intervals (2,000 query resamples, seed 42) compare methods within NFCorpus. Cross-dataset score differences are explicitly unpaired descriptive subtractions; the two datasets contain different queries and judgments. Four-method NFCorpus results are compared to the full 300-query SciFact Phase 2 run, while the router is compared to the 60-query SciFact Phase 3 internal slice. Those reference populations and dates remain separate. Existing SciFact latency numbers are historical observations; even on the same machine, runtime conditions and document lengths can differ. No causal or statistically controlled cross-domain timing conclusion is supported.

The hybrid-only router does not become adaptive merely by transferring. Matching hybrid quality and near-hybrid latency would demonstrate preserved behavior, not a learned generalization advantage. Whether its cost/quality compromise is sensible depends on explicit application requirements; the experiment does not invent one universal latency budget or retune the policy.

## Commands (Windows PowerShell, repository root)

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.transfer_policy
.\.venv\Scripts\python.exe -m app.data.nfcorpus
.\.venv\Scripts\python.exe -m app.retrieval.dense --dataset data/public/nfcorpus --cache indexes/nfcorpus-minilm.npz --batch-size 16 --threads 4
.\.venv\Scripts\python.exe -m app.evaluation.phase4
```

Existing installed neural/analysis dependencies are reused; a clean installation uses `python -m pip install -e '.[neural,analysis,dev]'` in the project virtual environment. A missing local model is handled with the exact pinned model download commands in README; no model was downloaded anew here.

The initial sandboxed NFCorpus request failed WinError 10061; the approved network retry with system HTTPS trust succeeded, without disabling certificate checks. If a direct download is blocked, keep the frozen protocol and use Windows trust:

```powershell
New-Item -ItemType Directory -Force data/public/archives | Out-Null
if (-not (Test-Path -LiteralPath data/public/archives/nfcorpus.zip)) {
    Invoke-WebRequest -Uri 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip' -OutFile data/public/archives/nfcorpus.zip -UseBasicParsing
}
.\.venv\Scripts\python.exe -m app.data.nfcorpus --archive data/public/archives/nfcorpus.zip
```

An interrupted/corrupt archive is rejected; download to a fresh filename and pass --archive rather than overwriting a verified ZIP. The mandatory policy-freeze command precedes this fallback too.

A Phase 4 run creates a new directory under experiments/results/phase4/. It contains all nine requested artifacts, extra raw live rows/rankings/candidate/analysis/routing files, copied exact frozen policy and protocol, and five figures generated from saved data. Failed runs retain explicit incomplete status. Replay validates input/output hashes, exact policy, identical query sets, rankings/metrics, decisions without labels, candidate provenance and aggregate/generalization reports without neural inference. The completed run is `phase4-20261031T071828Z-46027d7d4330`; measured results and limitations are in [experiment_results.md](experiment_results.md) and [the portable report](../experiments/reports/phase4_nfcorpus.json). Replay without model inference:

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.phase4 --dataset data/public/nfcorpus --replay experiments/results/phase4/phase4-20261031T071828Z-46027d7d4330
```

The full run evaluated all 323 queries. Frozen router share was 100% hybrid and 0% reranker invocation, matching hybrid metrics. Mean nDCG improved from BM25 to hybrid and from hybrid to reranked on this collection, but the additional neural path was substantially more expensive. This is a second-dataset observation with uncertainty and task/judgment differences, not a broad generalization proof. No policy threshold or model configuration was updated after observing results.
