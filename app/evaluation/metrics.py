import math
from collections.abc import Mapping, Sequence

from app.retrieval.base import validate_top_k


def ranking_metrics(
    ranking: Sequence[str], judgments: Mapping[str, int], k: int
) -> dict[str, float]:
    """Binary relevance >0; exponential graded nDCG; unjudged gain zero."""
    validate_top_k(k)
    if len(set(ranking)) != len(ranking):
        raise ValueError("Ranking contains duplicate document IDs")
    if any(type(g) is not int or not 0 <= g <= 100 for g in judgments.values()):
        raise ValueError("Grades must be integers between 0 and 100")
    relevant = {did for did, grade in judgments.items() if grade > 0}
    grades = [judgments.get(did, 0) for did in ranking[:k]]
    hits = sum(grade > 0 for grade in grades)
    rr = next((1 / rank for rank, grade in enumerate(grades, 1) if grade > 0), 0.0)

    def dcg(values: Sequence[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(values, 1))

    ideal = dcg(sorted(judgments.values(), reverse=True)[:k])
    return {
        f"recall@{k}": hits / len(relevant) if relevant else 0.0,
        f"precision@{k}": hits / k,
        f"mrr@{k}": rr,
        f"ndcg@{k}": dcg(grades) / ideal if ideal else 0.0,
    }
