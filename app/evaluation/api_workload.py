"""Seeded, label-free HTTP workload definitions and honest request aggregation."""

import csv
import ctypes
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

METHODS = ("bm25", "dense", "hybrid", "reranked", "router")


def queries(dataset: Path) -> list[dict]:
    """Fixed compatibility IDs from exposed Phase 2 examples; no qrels are opened."""
    public = {
        q["_id"]: q["text"]
        for q in (
            json.loads(line)
            for line in (dataset / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    result = [
        {"id": qid, "text": public[qid], "source": "BEIR-SciFact", "category": "public-claim"}
        for qid in ("94", "1241", "1088", "1024")
    ]
    for identifier, text, category in (
        ("authored-short", "vitamin D", "short-keywords"),
        (
            "authored-long",
            "What experimental evidence explains how inflammation changes immune cell responses in chronic disease?",
            "natural-language",
        ),
        ("authored-technical", "IL-6 TNF-alpha p53 2020 5%", "numbers-technical"),
        ("authored-punctuation", "Cancer: gene-expression (p53), cause or effect?", "punctuation"),
    ):
        result.append(
            {"id": identifier, "text": text, "source": "authored-not-judged", "category": category}
        )
    return result


def weights(value: str) -> dict[str, int]:
    result = {}
    for entry in value.split(","):
        name, number = entry.split(":")
        if name not in METHODS or name in result or int(number) <= 0:
            raise ValueError("Expected unique supported methods with positive integer weights")
        result[name] = int(number)
    return result


def schedule(pool: list[dict], count: int, distribution: dict[str, int], seed: int) -> list[dict]:
    if count < 1 or not pool or not distribution:
        raise ValueError("Nonempty query pool, positive count and method weights required")
    cycle = [m for m in METHODS for _ in range(distribution.get(m, 0))]
    rng = random.Random(seed)
    methods = [cycle[i % len(cycle)] for i in range(count)]
    rng.shuffle(methods)
    query_order = list(range(len(pool)))
    rng.shuffle(query_order)
    return [
        {"request_index": i, "method": method, "query": pool[query_order[i % len(pool)]]}
        for i, method in enumerate(methods)
    ]


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def aggregate(rows: list[dict], seconds: float) -> dict:
    def latency(values):
        if not values:
            return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
        return {
            "mean_ms": mean(values),
            **{f"p{p}_ms": float(np.percentile(values, p, method="linear")) for p in (50, 95, 99)},
        }

    successful = [r for r in rows if r["outcome"] == "success"]
    return {
        "requests": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "timed_out": sum(r["outcome"] == "timeout" for r in rows),
        "statuses": dict(
            Counter(str(r["status"]) if r["status"] is not None else r["outcome"] for r in rows)
        ),
        "elapsed_seconds": seconds,
        "attempts_per_second": len(rows) / seconds,
        "successful_per_second": len(successful) / seconds,
        "all_latency": latency([r["latency_ms"] for r in rows]),
        "success_latency": latency([r["latency_ms"] for r in successful]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an unspecified CSV schema")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def memory(pid: int) -> dict:
    """Native process working set/RSS; OS peak is lifetime, not a per-stage peak."""
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                    "PrivateUsage",
                )
            ]

        kernel, psapi = (
            ctypes.WinDLL("kernel32", use_last_error=True),
            ctypes.WinDLL("psapi", use_last_error=True),
        )
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        handle = kernel.OpenProcess(0x0410, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            counter = Counters()
            counter.cb = ctypes.sizeof(counter)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counter), counter.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            return {
                "rss_bytes": counter.WorkingSetSize,
                "os_lifetime_peak_rss_bytes": counter.PeakWorkingSetSize,
                "private_commit_bytes": counter.PrivateUsage,
            }
        finally:
            kernel.CloseHandle(handle)
    status = dict(
        line.split(":", 1)
        for line in Path(f"/proc/{pid}/status").read_text().splitlines()
        if ":" in line
    )
    return {
        "rss_bytes": int(status["VmRSS"].split()[0]) * 1024,
        "os_lifetime_peak_rss_bytes": int(status["VmHWM"].split()[0]) * 1024,
        "private_commit_bytes": None,
    }


class MemorySampler:
    def __init__(self, pid: int, interval: float = 0.1):
        self.pid, self.interval = pid, interval
        self.stage = "startup"
        self.start = time.perf_counter()
        self.rows = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self) -> None:
        try:
            self.rows.append(
                {
                    "elapsed_seconds": time.perf_counter() - self.start,
                    "stage": self.stage,
                    **memory(self.pid),
                }
            )
        except OSError as exc:
            self.errors.append(type(exc).__name__)

    def mark(self, stage: str) -> None:
        self.stage = stage
        self.sample()

    def run(self) -> None:
        while not self.stop.wait(self.interval):
            self.sample()

    def finish(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
