"""Five analytical figures derived only from saved Phase 3 data."""

import os
from pathlib import Path

from app.evaluation.phase3 import read_csv, read_json


def generate_plots(run_dir: Path) -> None:
    output = Path(run_dir) / "plots"
    output.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = read_csv(run_dir / "system_comparison.csv")
    evaluation = read_json(run_dir / "routing_evaluation.json")
    ndcg_key = next(key for key in summary[0] if key.startswith("ndcg@"))

    def save(name, title, xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output / name, dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    for row in summary:
        x, y = float(row["p50_latency_ms"]), float(row[ndcg_key])
        plt.scatter(x, y, label=row["method"])
        plt.annotate(
            row["method"],
            (x, y),
            xytext=(6, -18) if row["method"] == "router" else (6, 6),
            textcoords="offset points",
        )
    plt.xscale("log")
    save(
        "ndcg_vs_p50.png",
        "Held-out observed quality versus median latency",
        "p50 complete-query latency (ms, log scale)",
        ndcg_key,
    )
    comparisons = read_csv(run_dir / "per_query_comparison.csv")
    values = {(r["query_id"], r["method"]): float(r[ndcg_key]) for r in comparisons}
    delta = [values[q, "reranked"] - values[q, "hybrid"] for q, m in values if m == "hybrid"]
    plt.figure(figsize=(8, 5))
    plt.hist(delta, bins=[-1 + i * 0.1 for i in range(21)], edgecolor="white")
    save(
        "query_delta_distribution.png",
        "Historical Phase 2: reranked minus hybrid quality",
        "Per-query nDCG difference",
        "Query count",
    )
    failure = read_csv(run_dir / "failure_summary.csv")
    plt.figure(figsize=(10, 6))
    plt.barh([r["category"] for r in failure], [int(r["query_count"]) for r in failure])
    save(
        "failure_categories.png",
        "Nonexclusive evidence indicators (historical full 300 queries)",
        "Queries triggering indicator",
        "Indicator (inferred categories are hypotheses)",
    )
    routes = evaluation["route_fractions"]
    plt.figure(figsize=(8, 5))
    plt.bar(list(routes), [v * 100 for v in routes.values()])
    save(
        "router_distribution.png",
        "Frozen router: held-out strategy distribution",
        "Selected strategy",
        "Queries (%)",
    )
    plt.figure(figsize=(8, 5))
    for baseline, pairs in evaluation["paired_router_minus_baseline"].items():
        quality, latency = pairs[ndcg_key], pairs["latency_ms"]
        x, y = latency["mean_difference"], quality["mean_difference"]
        plt.errorbar(
            x,
            y,
            xerr=[[x - latency["ci95_lower"]], [latency["ci95_upper"] - x]],
            yerr=[[y - quality["ci95_lower"]], [quality["ci95_upper"] - y]],
            fmt="o",
            label=baseline,
        )
    plt.axhline(0, color="gray", linewidth=0.7)
    plt.axvline(0, color="gray", linewidth=0.7)
    plt.legend(title="Router minus baseline")
    save(
        "paired_quality_latency.png",
        "Exploratory paired query-bootstrap intervals (95%)",
        "Mean latency difference (ms; negative saves time)",
        "Mean nDCG difference (positive improves quality)",
    )
