"""Deterministic query-only subsets; the searchable corpus is never filtered."""

import hashlib

from app.data.schemas import Dataset


def select_queries(data: Dataset, max_queries: int | None = None, seed: int = 42) -> Dataset:
    if max_queries is not None and (type(max_queries) is not int or max_queries <= 0):
        raise ValueError("max_queries must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    ids = sorted(data.queries)
    if max_queries is not None:
        ids = sorted(
            ids, key=lambda qid: (hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest(), qid)
        )
        ids = sorted(ids[:max_queries])
    return Dataset(
        data.documents,
        {qid: data.queries[qid] for qid in ids},
        {qid: data.qrels[qid] for qid in ids},
        dict(data.metadata),
    )
