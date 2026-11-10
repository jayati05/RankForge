"""Exact deterministic partitions of an explicitly exposed benchmark."""

import hashlib


def exploratory_split(query_ids: list[str], seed: int = 42) -> dict:
    if len(set(query_ids)) != len(query_ids) or len(query_ids) < 5:
        raise ValueError("Require at least five unique query IDs")
    ordered = sorted(
        query_ids, key=lambda q: (hashlib.sha256(f"{seed}:{q}".encode()).hexdigest(), q)
    )
    n = len(ordered)
    development_end, validation_end = int(n * 0.6), int(n * 0.8)
    return {
        "seed": seed,
        "method": "SHA256 UTF-8 seed:query_id, ascending digest; 60/20/20 floor boundaries",
        "limitation": "Internal exploratory split of a previously inspected benchmark; not untouched external evaluation",
        "development": sorted(ordered[:development_end]),
        "validation": sorted(ordered[development_end:validation_end]),
        "held_out": sorted(ordered[validation_end:]),
    }


def validate_partition(split: dict, universe: list[str]) -> None:
    groups = [split[name] for name in ("development", "validation", "held_out")]
    flat = [q for group in groups for q in group]
    if (
        any(not group for group in groups)
        or len(flat) != len(set(flat))
        or set(flat) != set(universe)
    ):
        raise ValueError("Split must cover every query exactly once with three nonempty groups")
