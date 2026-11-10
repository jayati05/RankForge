# RankForge architecture

## Phase 7A container boundary

The existing single-worker API runs as non-root PID1 inside a Linux/amd64 CPU image. Docker copies source and pinned dependencies only; corpus, model snapshots, separate Linux dense cache and frozen policy enter through read-only bind mounts. Container-only required-file preflight precedes the unchanged lazy lifecycle. SIGTERM reaches Uvicorn directly. Health calls the existing model-free endpoint; explicit integration verification tests all-method readiness, equivalence, limits and restart. The host verifier writes experiment artifacts. See [Docker guide](docker_deployment.md) for paths, defaults and measured limitations.

## Phase 1 architecture

```mermaid
flowchart LR
  A[JSONL corpus and queries + TSV qrels] --> B[Validated dataset loader]
  B --> C[Shared Unicode tokenizer]
  C --> D[BM25 Okapi index]
  D --> E[Typed ranked document results]
  B --> F[Ranking metrics]
  E --> F
  F --> G[Unique run directory: CSV and JSON]
```

The benchmark CLI composes independent loading, retrieval, and metric modules. Index construction occurs once per run, outside query timing. The following sections document subsequent extensions.

## Phase 2 extensions

```mermaid
flowchart LR
  D[Validated shared corpus] --> B[Unchanged BM25 index]
  D --> E[SentenceTransformer document encoder]
  E --> C[Validated normalized vector cache]
  Q[Query] --> B
  Q --> V[SentenceTransformer query encoder]
  V --> N[NumPy cosine search]
  C --> N
  B --> F[Equal-weight RRF]
  N --> F
  F --> R[Top-20 cross-encoder pairs]
  R --> T[Reranked top-10]
  B --> M[Unchanged metrics]
  N --> M
  F --> M
  T --> M
  M --> A[Comparison artifacts and replay]
```

`app.retrieval.base.Retriever` remains the common search contract. Neural dependencies are imported only by model adapters when initialized. The original single-BM25 benchmark CLI remains available; `app.evaluation.comparison` composes all four strategies with independent timed searches. Later phases add frozen routing and the local API below; no dashboard or container is included.


## Phase 5 service boundary

```mermaid
flowchart LR
  H[HTTP requests / Pydantic contracts] --> B[Request IDs, bounded bodies, safe errors]
  B --> S[Search / comparison services]
  S --> L[Lazy resource manager and bounded execution lock]
  L --> C[Corpus-only loader]
  L --> R[Existing BM25 / dense / RRF / reranker]
  L --> F[Exact frozen router and corpus features]
  R --> P[Ranked response payload]
  F --> P
  Z[Health and models] --> T[Short status lock, presence checks only]
  E[Independent offline evaluation CLIs] --> A[Existing qrels / metrics / experiment artifacts]
```

Lifespan owns one manager per process. Local prepared neural adapters and read-only validated vector caches are reused; lazy retriever proxies let the frozen hybrid-only router avoid loading the cross-encoder. No retriever imports FastAPI. API loading uses only corpus.jsonl, and there is no HTTP evaluation/write/download endpoint. A separate status lock keeps health responsive during initialization. Resource setup, complete retrieval and queue waiting are separately measured. The [service guide](api_service.md) describes schemas, concurrency/error boundaries and limitations; prior evaluation architecture remains independent.
