"""Five external-validation figures from saved comparison and reference data."""

import os
from pathlib import Path

from app.evaluation.phase3 import read_csv, read_json


def generate_plots(root: Path) -> None:
    output = Path(root) / "plots"
    output.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = read_csv(root / "system_comparison.csv")
    gen = read_json(root / "generalization_analysis.json")
    key = next(k for k in rows[0] if k.startswith("ndcg@"))

    def save(name, title, xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output / name, dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    for row in rows:
        x, y = float(row["p50_latency_ms"]), float(row[key])
        plt.scatter(x, y)
        plt.annotate(
            row["method"],
            (x, y),
            xytext=(5, -18) if row["method"] == "router" else (5, 6),
            textcoords="offset points",
        )
    plt.xscale("log")
    save(
        "ndcg_vs_p50.png",
        "NFCorpus: observed quality versus median latency",
        "p50 complete-query latency (ms, log scale)",
        key,
    )
    plt.figure(figsize=(9, 5))
    metric_keys = [
        next(k for k in rows[0] if k.startswith(name + "@"))
        for name in ("recall", "precision", "mrr", "ndcg")
    ]
    x = np.arange(len(rows))
    width = 0.2
    for i, m in enumerate(metric_keys):
        plt.bar(x + (i - 1.5) * width, [float(r[m]) for r in rows], width, label=m)
    plt.xticks(x, [r["method"] for r in rows])
    plt.legend()
    save(
        "retrieval_quality.png", "NFCorpus: same-query retrieval quality", "Method", "Metric value"
    )
    comparisons = read_csv(root / "per_query_comparison.csv")
    values = {(r["query_id"], r["method"]): float(r[key]) for r in comparisons}
    delta = [values[q, "reranked"] - values[q, "hybrid"] for q, m in values if m == "hybrid"]
    plt.figure(figsize=(8, 5))
    plt.hist(delta, bins=np.linspace(-1, 1, 21), edgecolor="white")
    save(
        "reranker_delta_distribution.png",
        "NFCorpus: query-level reranked minus hybrid quality",
        "nDCG difference",
        "Query count",
    )
    nf = gen["nfcorpus_failure_summary"]
    sf = {r["category"]: float(r["query_fraction"]) for r in gen["scifact_full_failure_summary"]}
    y = np.arange(len(nf))
    plt.figure(figsize=(11, 6))
    plt.barh(
        y - 0.2,
        [float(r["query_fraction"]) * 100 for r in nf],
        0.4,
        label=f"NFCorpus ({gen['nfcorpus_query_count']})",
    )
    plt.barh(
        y + 0.2,
        [sf[r["category"]] * 100 for r in nf],
        0.4,
        label=f"SciFact ({gen['scifact_phase2_query_count']})",
    )
    plt.yticks(y, [r["category"] for r in nf])
    plt.legend()
    save(
        "failure_category_comparison.png",
        "Nonexclusive indicators: different judgment density and tasks",
        "Queries triggering indicator (%)",
        "Evidence indicator (inferred categories are hypotheses)",
    )
    sf2 = {r["method"]: r for r in gen["scifact_phase2_full_summary"]}
    nfmap = {r["method"]: r for r in gen["nfcorpus_summary"]}
    methods = list(sf2)
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, m in zip(axes.flat, metric_keys):
        axis.bar(x - 0.18, [sf2[n][m] for n in methods], 0.36, label="SciFact full")
        axis.bar(x + 0.18, [nfmap[n][m] for n in methods], 0.36, label="NFCorpus test")
        axis.set_xticks(x, methods, rotation=15)
        axis.set_ylabel(m)
        axis.legend()
    fig.suptitle("Cross-dataset descriptive scores: different queries, grades and judgment density")
    fig.tight_layout()
    fig.savefig(output / "scifact_nfcorpus_metrics.png", dpi=160)
    plt.close(fig)
