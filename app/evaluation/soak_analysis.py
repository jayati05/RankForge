"""Memory windows and explicit local budgets, without inferring a leak from RSS."""

import numpy as np


def slope(rows: list[dict]) -> float | None:
    if len(rows) < 2 or rows[-1]["elapsed_seconds"] <= rows[0]["elapsed_seconds"]:
        return None
    x = np.array([r["elapsed_seconds"] for r in rows]) / 60
    y = np.array([r["rss_bytes"] for r in rows]) / 2**20
    return float(np.polyfit(x - x[0], y, 1)[0])


def analyze(rows: list[dict], stage: str, window_seconds: float, growth_budget: float) -> dict:
    active = [r for r in rows if r["stage"] == stage]
    if not active:
        return {"classification": "insufficient_samples", "sample_count": 0}
    start, end = active[0]["elapsed_seconds"], active[-1]["elapsed_seconds"]
    early = [r for r in active if r["elapsed_seconds"] <= start + window_seconds]
    late = [r for r in active if r["elapsed_seconds"] >= end - window_seconds]
    enough = end - start >= 2 * window_seconds and len(late) >= 3
    late_slope = slope(late)
    classification = "insufficient_duration"
    if enough:
        classification = (
            "increasing"
            if late_slope > growth_budget
            else "decreasing"
            if late_slope < -growth_budget
            else "approximately_flat_late_window"
        )
    idle = [r for r in rows if r["stage"] == stage + "_idle"]
    return {
        "stage": stage,
        "sample_count": len(active),
        "observed_span_seconds": end - start,
        "window_seconds": window_seconds,
        "whole_slope_mib_per_minute": slope(active),
        "late_slope_mib_per_minute": late_slope,
        "classification": classification,
        "early_mean_mib": float(np.mean([r["rss_bytes"] for r in early])) / 2**20,
        "late_mean_mib": float(np.mean([r["rss_bytes"] for r in late])) / 2**20,
        "peak_mib": max(r["rss_bytes"] for r in active) / 2**20,
        "end_mib": active[-1]["rss_bytes"] / 2**20,
        "idle_end_mib": idle[-1]["rss_bytes"] / 2**20 if idle else None,
        "idle_change_mib": (idle[-1]["rss_bytes"] - active[-1]["rss_bytes"]) / 2**20
        if idle
        else None,
        "growth_budget_passed": late_slope <= growth_budget if enough else None,
        "interpretation": "RSS trend/recovery observations only; native retention versus leaks require allocation evidence. Flat late windows do not prove bounded lifetime memory.",
    }


def budgets(
    stats: dict, memory: dict, rss_limit: float, rejection_limit: float, p95_limit: float
) -> dict:
    requests = stats["requests"]
    rejected = stats["statuses"].get("429", 0)
    other = stats["failed"] - rejected - stats["timed_out"] - stats.get("cancelled", 0)
    checks = {
        "rejection_rate": rejected / requests <= rejection_limit if requests else False,
        "timeouts": stats["timed_out"] == 0,
        "cancellations": stats.get("cancelled", 0) == 0,
        "other_errors": other == 0,
        "p95_latency": stats["success_latency"]["p95_ms"] <= p95_limit
        if stats["successful"]
        else False,
        "rss_peak": memory.get("peak_mib", float("inf")) <= rss_limit,
        "late_memory_growth": memory.get("growth_budget_passed"),
    }
    return {
        "checks": checks,
        "all_passed": all(v is True for v in checks.values()),
        "indeterminate": any(v is None for v in checks.values()),
        "rejection_rate": rejected / requests if requests else None,
        "other_errors": other,
    }
