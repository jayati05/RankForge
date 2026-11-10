# Data provenance and format

`smoke/` is SYNTHETIC, authored for RankForge software verification: 6 documents, 4 queries, 4 positive judgments. Version 1. No external text is included. Do not combine its results with research benchmarks.

Loader layout follows BEIR:
- `corpus.jsonl`: `_id` string, optional `title` string (default empty), `text` string.
- `queries.jsonl`: `_id` string and nonempty `text` string.
- `qrels/<split>.tsv`: exact header `query-id`, `corpus-id`, `score` separated by tabs; nonnegative integer grades (0..100).
- Optional `metadata.json`: provenance JSON object, copied into each run.

Duplicate IDs/pairs, malformed records, missing joins and empty splits are rejected. Only queries appearing in the selected qrels split are evaluated; all corpus documents remain searchable.

Sources inspected 2026-10-25:
- BEIR format and dataset overview: https://github.com/beir-cellar/beir
- SciFact dataset card: https://huggingface.co/datasets/BeIR/scifact (lists CC-BY-SA-4.0; corpus `_id`, `title`, `text`).
- Original project: https://github.com/allenai/scifact

## Verified SciFact workflow

The public-data milestone downloaded the official [BEIR SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) after reviewing the source and license. The [BEIR dataset table](https://github.com/beir-cellar/beir#beers-available-datasets) publishes MD5 `5f7d1de60b170fc8027bb7898e2efca1`, which the importer requires. SHA256 of the verified archive is `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.

The documented [BEIR loader](https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py) confirms the JSONL/TSV layout used here. The [SciFact card](https://huggingface.co/datasets/BeIR/scifact) lists CC-BY-SA-4.0. [License terms](https://creativecommons.org/licenses/by-sa/4.0/) require appropriate attribution, a license link and change notices, with ShareAlike for distributed adaptations and no additional restrictions. Credit SciFact (Wadden et al., EMNLP 2020) and BEIR (Thakur et al., 2021). This workflow downloads public material locally without publishing data or imposing additional terms.

`python -m app.data.download` uses HTTPS and verifies the archive. `--archive <path>` imports an already-downloaded official ZIP using the same checks. Four files (corpus, queries, train qrels, test qrels) are copied byte-for-byte; no arbitrary ZIP members are extracted. Both splits are validated before the staged dataset is renamed into place. `metadata.json` records exact provenance, hashes, observed counts and changes. Existing verified data can be reused offline; differing files or existing unrelated destinations are rejected. There is no overwrite option.

Observed counts: 5,183 documents; BEIR train 809 queries / 919 judgments; BEIR test 300 queries / 339 judgments. Corpus is shared between splits, with disjoint judged query IDs. This is BEIR's retrieval conversion, not a direct import of original SciFact claim-verification files. No relevance labels were invented.

All external artifacts are under ignored `data/public/`. Use the exact verified Windows PowerShell commands in [README](../README.md#public-scifact-benchmark-windows-powershell). On this machine direct Python HTTPS failed `CERTIFICATE_VERIFY_FAILED`; PowerShell Invoke-WebRequest succeeded using Windows trust. The importer then verified the published checksum. No TLS checks were disabled. Retry a network failure with those same PowerShell download/import commands; use a fresh archive filename if a previous download was interrupted.
