from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    rank: int
    score: float


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]: ...


def validate_top_k(top_k: int) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
