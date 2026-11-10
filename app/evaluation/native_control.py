"""Isolated, authored-data OpenBLAS import/kernel control (not an IR benchmark)."""

import argparse
import json
import os
from pathlib import Path
import time

from app.evaluation.windows_native import counters, regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshots = []

    def snapshot(label):
        snapshots.append(
            {
                "label": label,
                "seconds": time.perf_counter() - started,
                **counters(os.getpid()),
                "regions": regions(os.getpid())["totals"],
            }
        )

    started = time.perf_counter()
    snapshot("before_numpy_import")
    import numpy as np
    from threadpoolctl import threadpool_info

    snapshot("after_numpy_import")
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((5183, 384), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.standard_normal(384, dtype=np.float32)
    query /= np.linalg.norm(query)
    snapshot("after_authored_arrays")
    first = matrix @ query
    for _ in range(100):
        result = matrix @ query
        if not np.allclose(first, result, rtol=0, atol=1e-6):
            raise AssertionError("Kernel output changed")
    snapshot("after_100_kernel_calls")
    np.save(args.output / "scores.npy", result)
    report = {
        "pid": os.getpid(),
        "status": "complete",
        "snapshots": snapshots,
        "threadpools": threadpool_info(),
        "environment": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "array_bytes": matrix.nbytes + query.nbytes,
        "versions": {"numpy": np.__version__},
        "design": "Only OPENBLAS_NUM_THREADS differs across fresh processes1/20; seed42, same float32 normalized authored5183x384 matrix/vector and100calls. No IR judgments/models. Private/RSS snapshots are not native allocation stacks; no GC/allocator release.",
    }
    (args.output / "control.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pid": os.getpid(), "status": "complete"}))


if __name__ == "__main__":
    main()
