# Ranking Failure Lab

Phase 3 extends the existing Phase 2 persistence, data schema, tokenizer and metrics. It does not rerun historical benchmarks merely to recalculate metrics. The source comparison is replay-verified before analysis; its files are hashed and never overwritten. Historical failure tables include all source query IDs (300 in the public full run), all four systems, original timings, and supplied qrels.

## Query comparisons and judgment conventions

`per_query_comparison.csv` records Recall/Precision/MRR/nDCG at the source cutoff, first judged-positive rank within that cutoff, relevant hits/misses, known reranker-input pool size, total pipeline latency, incremental reranking time, and retrieval latency estimated by subtracting incremental reranking time. This subtraction is a component estimate, not a separately timed independent retrieval. The pool size for hybrid/reranked refers to the saved top-20 reranker-input pool, not the full union of the two depth-50 RRF source lists. Standalone methods have no saved reranker pool (blank), rather than an invented count. Each method is labeled improved/degraded/unchanged against BM25; JSON also includes all ordered pairwise nDCG differences and effects.

Missing/empty judgments are not discarded by the analysis function: metrics are zero under the existing tested definitions and judgment_status explicitly distinguishes no_judgments from no_positive_judgments. The public analysis universe is the exact Phase 2 query IDs; it does not silently add other SciFact splits. The current source has no zero-judgment queries. Unit tests cover them. Unjudged documents have zero evaluation gain; they are not called known irrelevant. First relevant rank is bounded by the saved top-k; null does not mean absent from the corpus.

## Nonexclusive taxonomy

| Indicator | Evidence required | Interpretation |
| --- | --- | --- |
| lexical_mismatch | A judged-positive document absent from BM25 top-k has zero shared lexical tokens with query | Inferred indicator; no causal claim |
| semantic_mismatch | A judged-positive document appears in BM25 top-k but not dense top-k | Differential retrieval is observed; semantic failure is only a hypothesis |
| candidate_generation_failure | Judged-positive ID absent from the saved hybrid reranker-input pool | Observed; reranking cannot recover that ID |
| ranking_failure | Judged-positive ID in that pool, below hybrid top-k | Observed candidate/rank boundary |
| reranker_regression | Reranked nDCG below hybrid nDCG | Observed query-level metric degradation |
| ambiguous_or_underspecified | Query has at most three lexical tokens | Short-query heuristic, not ground-truth ambiguity |
| insufficient_judgments | At most one positive judgment (including none) | Evidence-sparsity warning; not a claim of incorrect labels |
| other_or_unclassified | No defined rule fires | Fallback; does not assert failure or success |

Each classification stores evidence_type and the exact supporting counts/IDs/metric values. Multiple indicators can apply, so category counts need not sum to query count. No query IDs or manual failure labels are embedded in the rules. The JSON saves every query, including queries without cross-system disagreement. Categories are deliberately conservative diagnostics; the system does not establish a model's causal reason for failing.

## Label-free feature definitions

The reusable corpus feature extractor accepts documents only and never accepts qrels. It uses the unchanged NFKC/casefold identifier tokenizer. Document frequency counts each token at most once per document. Rare means 0 < document frequency <= max(1, ceil(0.01 * corpus document count)); out-of-vocabulary tokens are uncovered rather than called rare.

| Feature | Definition |
| --- | --- |
| token_count | Number of lexical tokens including repeats |
| unique_token_count | Distinct lexical tokens |
| average_token_length | Mean token character length including repeats; zero for empty |
| numeric_indicator | Any Unicode digit in raw query |
| punctuation_indicator | Any raw character that is neither alphanumeric nor whitespace; includes symbols |
| rare_term_proportion | Rare distinct query tokens divided by all distinct query tokens; zero if empty |
| corpus_term_coverage | Distinct query tokens present in corpus divided by all distinct query tokens; zero if empty |
| bm25/dense score mean/std/max/gap | Saved score population mean/std/max and first-minus-second gap; missing statistic is blank |
| ranking_position_agreement | Identical IDs at the same rank divided by larger saved list length |
| topk_jaccard | Intersection divided by union of saved BM25/dense ID sets |
| candidate_pool_overlap_topk | Intersection divided by the smaller saved BM25/dense list size |

The score/overlap features are diagnostics from the source's saved top-k lists (depths saved alongside). They are explicitly not the complete depth-50 candidate-source overlap, which Phase 2 did not persist. They are not inputs to the cheap router and cannot be used for free without executing their associated retrievers. Corpus-statistics construction is startup work, excluded from per-query latency; actual live router latency includes extraction and decision.

## Artifacts and figures

`failure_cases.csv` has one query/indicator row with JSON evidence. `failure_analysis.json` saves all query analyses and pairwise outcomes. `failure_summary.csv` contains every category, including zero-count categories. `query_features.csv` holds label-free and saved-ranking diagnostics. `per_query_comparison.csv` holds historical source metrics/timings. Original source IDs, configuration and artifact hashes remain in run_metadata.json.

Five Matplotlib figures are generated from saved Phase 3 files: observed nDCG versus p50 latency, historical per-query rerank quality differences, nonexclusive indicator counts, live router route shares, and paired router-minus-baseline quality/latency confidence intervals. Historical full-300-query plots are labeled separately from fresh internal-held-out timing plots. No scores are manually entered into figures.

See [routing design](query_aware_routing.md), [measured results](experiment_results.md), and README for commands. No API/dashboard layer is implemented here.
